"""Streamlit web UI for PromptDiff."""

import json
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

from promptdiff.runner import run_diff
from promptdiff.comparison import compare_multiple_models
from promptdiff.report.markdown import generate as generate_markdown


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PromptDiff",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 PromptDiff")
    st.markdown("**Git-style diffs for LLM outputs**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Comparison Mode",
            ["Single Comparison", "Multi-Model Comparison"],
            help="Compare two models or multiple models pairwise"
        )
        
        st.divider()
        
        # Prompts file upload/selection
        st.subheader("📝 Prompts")
        prompt_source = st.radio(
            "Prompt Source",
            ["Type/ Paste JSON", "Upload File", "Use Example", "Add Manually"],
            horizontal=False
        )
        
        prompts_data = None
        
        if prompt_source == "Type/ Paste JSON":
            st.info("💡 Paste your prompts JSON below. Format: array of objects with 'id', 'prompt', and 'vars' fields.")
            json_text = st.text_area(
                "Prompts JSON",
                height=300,
                placeholder='[\n  {\n    "id": "test1",\n    "prompt": "Summarize: {{text}}",\n    "vars": {"text": "Your text here"}\n  }\n]',
                help="Enter prompts as a JSON array"
            )
            if json_text.strip():
                try:
                    prompts_data = json.loads(json_text)
                    if isinstance(prompts_data, list):
                        st.success(f"✓ Loaded {len(prompts_data)} prompt(s)")
                    else:
                        st.error("JSON must be an array of prompt objects")
                        prompts_data = None
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
                    prompts_data = None
                except Exception as e:
                    st.error(f"Error parsing JSON: {e}")
                    prompts_data = None
        
        elif prompt_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload prompts JSON",
                type=["json"],
                help="Upload a JSON file with prompts"
            )
            if uploaded_file:
                try:
                    prompts_data = json.load(uploaded_file)
                    if isinstance(prompts_data, list):
                        st.success(f"✓ Loaded {len(prompts_data)} prompt(s)")
                    else:
                        st.error("File must contain a JSON array")
                        prompts_data = None
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        elif prompt_source == "Use Example":
            # Load example prompts and let user select
            example_path = Path(__file__).parent.parent / "examples" / "prompts.json"
            if example_path.exists():
                with open(example_path, "r", encoding="utf-8") as f:
                    all_examples = json.load(f)
                
                # Create a list of prompt IDs for selection
                prompt_options = {f"{p.get('id', 'unnamed')}": p for p in all_examples}
                
                st.write(f"**Select example prompts ({len(all_examples)} available):**")
                
                # Multi-select for examples
                selected_ids = st.multiselect(
                    "Choose prompts to use",
                    options=list(prompt_options.keys()),
                    default=[],  # No defaults - user must select
                    help="Select one or more example prompts to use"
                )
                
                if selected_ids:
                    prompts_data = [prompt_options[pid] for pid in selected_ids]
                    st.success(f"✓ Selected {len(prompts_data)} prompt(s)")
                else:
                    st.info("👆 Select at least one prompt from the list above")
                    prompts_data = None
            else:
                st.warning("Example prompts file not found")
        
        elif prompt_source == "Add Manually":
            prompts_data = _manual_prompt_builder()
        
        st.divider()
        
        # Model configuration - moved here so it's always visible
        st.subheader("🤖 Model Configuration")
        
        baseline_provider = st.selectbox(
            "Baseline Provider",
            ["openai", "anthropic", "ollama", "local"],
            key="sidebar_baseline_provider"
        )
        baseline_model = st.text_input(
            "Baseline Model Name",
            value="gpt-4" if baseline_provider == "openai" else "llama3",
            key="sidebar_baseline_model",
            help="e.g., 'gpt-4', 'claude-3-opus', 'llama3'"
        )
        
        candidate_provider = st.selectbox(
            "Candidate Provider",
            ["openai", "anthropic", "ollama", "local"],
            index=2 if baseline_provider == "openai" else 0,
            key="sidebar_candidate_provider"
        )
        candidate_model = st.text_input(
            "Candidate Model Name",
            value="llama3" if candidate_provider == "ollama" else "gpt-4.1",
            key="sidebar_candidate_model",
            help="e.g., 'gpt-4.1', 'claude-3-sonnet', 'granite4'"
        )
        
        st.divider()
        
        # Model parameters
        st.subheader("🔧 Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in model output"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=4096,
            value=1024,
            step=100,
            help="Maximum tokens in model response"
        )
        
        embedding_model = st.text_input(
            "Embedding Model (Optional)",
            value="",
            placeholder="sentence-transformers/all-MiniLM-L6-v2",
            help="Custom embedding model for semantic diff"
        )
    
    # Main content area
    if not prompts_data:
        st.info("👈 Please load prompts from the sidebar to get started")
        _show_example_prompt_format()
        return
    
    # Display prompts preview
    with st.expander("📋 View Prompts", expanded=False):
        st.json(prompts_data)
    
    if mode == "Single Comparison":
        _single_comparison_ui(prompts_data, temperature, max_tokens, embedding_model)
    else:
        _multi_comparison_ui(prompts_data, temperature, max_tokens, embedding_model)


def _single_comparison_ui(prompts_data: List[Dict], temperature: float, max_tokens: int, embedding_model: str):
    """UI for single model pair comparison."""
    st.header("🔍 Single Model Comparison")
    
    # Get model info from sidebar session state
    baseline_provider = st.session_state.get("sidebar_baseline_provider", "openai")
    baseline_model = st.session_state.get("sidebar_baseline_model", "gpt-4")
    candidate_provider = st.session_state.get("sidebar_candidate_provider", "ollama")
    candidate_model = st.session_state.get("sidebar_candidate_model", "llama3")
    
    # Display current model selection
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Baseline:** {baseline_provider}:{baseline_model}")
    with col2:
        st.info(f"**Candidate:** {candidate_provider}:{candidate_model}")
    
    st.caption("💡 Change models in the sidebar configuration")
    
    # Build model identifiers
    baseline = f"{baseline_provider}:{baseline_model}" if baseline_provider != "openai" or ":" in baseline_model else baseline_model
    candidate = f"{candidate_provider}:{candidate_model}" if candidate_provider != "openai" or ":" in candidate_model else candidate_model
    
    # Save prompts to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(prompts_data, f)
        temp_prompts_file = f.name
    
    if st.button("🚀 Run Comparison", type="primary", width='stretch'):
        with st.spinner("Running comparison... This may take a while."):
            try:
                model_kwargs = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                results = run_diff(
                    prompts_file=temp_prompts_file,
                    baseline=baseline,
                    candidate=candidate,
                    embedding_model=embedding_model if embedding_model else None,
                    **model_kwargs
                )
                
                st.session_state['results'] = results
                st.session_state['baseline'] = baseline
                st.session_state['candidate'] = candidate
                st.success("✅ Comparison complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)
    
    # Display results
    if 'results' in st.session_state:
        _display_results(
            st.session_state['results'],
            st.session_state.get('baseline', 'baseline'),
            st.session_state.get('candidate', 'candidate')
        )


def _multi_comparison_ui(prompts_data: List[Dict], temperature: float, max_tokens: int, embedding_model: str):
    """UI for multi-model comparison."""
    st.header("🔬 Multi-Model Comparison")
    
    st.info("Compare multiple models pairwise. Each model will be compared with every other model.")
    
    # Initialize models list
    if 'multi_models' not in st.session_state:
        st.session_state.multi_models = []
    
    # Quick add from sidebar models
    st.subheader("Quick Add from Sidebar")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Baseline Model", width='stretch'):
            baseline_provider = st.session_state.get("sidebar_baseline_provider", "openai")
            baseline_model = st.session_state.get("sidebar_baseline_model", "gpt-4")
            model_id = f"{baseline_provider}:{baseline_model}" if baseline_provider != "openai" or ":" in baseline_model else baseline_model
            st.session_state.multi_models.append((baseline_model, model_id))
            st.rerun()
    with col2:
        if st.button("➕ Add Candidate Model", width='stretch'):
            candidate_provider = st.session_state.get("sidebar_candidate_provider", "ollama")
            candidate_model = st.session_state.get("sidebar_candidate_model", "llama3")
            model_id = f"{candidate_provider}:{candidate_model}" if candidate_provider != "openai" or ":" in candidate_model else candidate_model
            st.session_state.multi_models.append((candidate_model, model_id))
            st.rerun()
    
    st.divider()
    
    # Model input
    st.subheader("Configure Models")
    num_models = st.number_input(
        "Number of Models",
        min_value=max(2, len(st.session_state.multi_models)),
        max_value=10,
        value=max(3, len(st.session_state.multi_models)),
        step=1
    )
    
    # Ensure session state has enough models
    while len(st.session_state.multi_models) < num_models:
        st.session_state.multi_models.append((f"model{len(st.session_state.multi_models)+1}", f"model{len(st.session_state.multi_models)+1}"))
    
    models = []
    cols = st.columns(min(3, num_models))
    
    for i in range(num_models):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.subheader(f"Model {i+1}")
            
            # Get existing model if available
            existing_display, existing_id = st.session_state.multi_models[i] if i < len(st.session_state.multi_models) else (f"model{i+1}", f"model{i+1}")
            
            provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "ollama", "local"],
                key=f"model_{i}_provider"
            )
            model_name = st.text_input(
                "Model Name",
                value=existing_display,
                key=f"model_{i}_name"
            )
            display_name = st.text_input(
                "Display Name (Optional)",
                value=existing_display,
                key=f"model_{i}_display"
            )
            
            model_id = f"{provider}:{model_name}" if provider != "openai" or ":" in model_name else model_name
            models.append((display_name, model_id))
            
            # Update session state
            if i < len(st.session_state.multi_models):
                st.session_state.multi_models[i] = (display_name, model_id)
    
    # Display current models
    if st.session_state.multi_models:
        st.divider()
        st.write("**Current Models:**")
        for i, (display, model_id) in enumerate(st.session_state.multi_models[:num_models]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{i+1}. {display} ({model_id})")
            with col2:
                if st.button("🗑️", key=f"remove_model_{i}"):
                    st.session_state.multi_models.pop(i)
                    st.rerun()
    
    # Save prompts to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(prompts_data, f)
        temp_prompts_file = f.name
    
    if st.button("🚀 Run Multi-Model Comparison", type="primary", width='stretch'):
        with st.spinner("Running comparisons... This may take a while."):
            try:
                model_kwargs = {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                result = compare_multiple_models(
                    prompts_file=temp_prompts_file,
                    models=models,
                    embedding_model=embedding_model if embedding_model else None,
                    **model_kwargs
                )
                
                st.session_state['multi_results'] = result
                st.success("✅ All comparisons complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)
    
    # Display results
    if 'multi_results' in st.session_state:
        _display_multi_results(st.session_state['multi_results'])


def _display_results(results: List[Dict], baseline: str, candidate: str):
    """Display single comparison results."""
    st.header("📊 Results")
    
    # Summary stats
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Prompts", len(results))
    with col2:
        st.metric("Successful", len(successful))
    with col3:
        st.metric("Failed", len(failed))
    with col4:
        if successful:
            avg_sim = sum(r.get("semantic_similarity", 0) for r in successful) / len(successful)
            st.metric("Avg Similarity", f"{avg_sim:.3f}")
    
    st.divider()
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📈 Summary", "📄 Report"])
    
    with tab1:
        for i, result in enumerate(results):
            with st.expander(f"Prompt: {result.get('id', f'Prompt {i+1}')}", expanded=i == 0):
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"Baseline ({baseline})")
                        st.text_area(
                            "Output",
                            value=result.get("baseline_output", ""),
                            height=200,
                            key=f"baseline_{i}",
                            label_visibility="collapsed"
                        )
                    
                    with col2:
                        st.subheader(f"Candidate ({candidate})")
                        st.text_area(
                            "Output",
                            value=result.get("candidate_output", ""),
                            height=200,
                            key=f"candidate_{i}",
                            label_visibility="collapsed"
                        )
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Semantic Similarity", f"{result.get('semantic_similarity', 0):.3f}")
                    with col2:
                        text_stats = result.get("text_stats", {})
                        st.metric("Text Similarity", f"{text_stats.get('similarity', 0):.3f}")
                    with col3:
                        st.metric("Length Diff", f"{text_stats.get('length_diff', 0)} chars")
                    
                    # Text diff
                    if result.get("text_diff"):
                        with st.expander("View Text Diff"):
                            st.code(result["text_diff"], language="diff")
    
    with tab2:
        if successful:
            similarities = [r.get("semantic_similarity", 0) for r in successful]
            import pandas as pd
            
            df = pd.DataFrame({
                "Prompt ID": [r.get("id", "unknown") for r in successful],
                "Similarity": similarities,
                "Baseline Length": [r.get("text_stats", {}).get("baseline_length", 0) for r in successful],
                "Candidate Length": [r.get("text_stats", {}).get("candidate_length", 0) for r in successful],
            })
            st.dataframe(df, width='stretch')
            
            # Chart
            st.bar_chart(df.set_index("Prompt ID")["Similarity"])
    
    with tab3:
        report = generate_markdown(results)
        st.markdown(report)
        
        # Download button
        st.download_button(
            label="Download Report",
            data=report,
            file_name="promptdiff_report.md",
            mime="text/markdown"
        )
        
        # Download JSON
        json_data = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Results JSON",
            data=json_data,
            file_name="promptdiff_results.json",
            mime="application/json"
        )


def _display_multi_results(result: Dict[str, Any]):
    """Display multi-model comparison results."""
    st.header("📊 Multi-Model Comparison Results")
    
    summary = result.get("summary", {})
    results = result.get("results", [])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comparisons", summary.get("total_comparisons", 0))
    with col2:
        st.metric("Successful", summary.get("successful", 0))
    with col3:
        st.metric("Failed", summary.get("failed", 0))
    with col4:
        st.metric("Avg Similarity", f"{summary.get('avg_similarity', 0):.3f}")
    
    st.divider()
    
    # Report
    tab1, tab2 = st.tabs(["📄 Report", "📋 Raw Results"])
    
    with tab1:
        report = result.get("report", "")
        if report:
            st.markdown(report)
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="multi_model_comparison_report.md",
                mime="text/markdown"
            )
        else:
            st.warning("No report generated")
    
    with tab2:
        st.json(results)
        
        json_data = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Results JSON",
            data=json_data,
            file_name="multi_model_comparison_results.json",
            mime="application/json"
        )


def _manual_prompt_builder() -> Optional[List[Dict]]:
    """Manual prompt builder interface."""
    st.info("💡 Add prompts one by one using the form below")
    
    if 'manual_prompts' not in st.session_state:
        st.session_state.manual_prompts = []
    
    with st.form("add_prompt_form", clear_on_submit=True):
        st.subheader("Add New Prompt")
        
        prompt_id = st.text_input("Prompt ID *", placeholder="summary_test")
        prompt_template = st.text_area(
            "Prompt Template *",
            placeholder='Summarize this article in 3 bullet points: {{text}}',
            height=100,
            help="Use {{variable}} syntax for variables"
        )
        
        # Variables input
        st.write("**Variables (Optional)**")
        num_vars = st.number_input("Number of Variables", min_value=0, max_value=10, value=0)
        
        vars_dict = {}
        for i in range(num_vars):
            col1, col2 = st.columns(2)
            with col1:
                var_name = st.text_input(f"Variable {i+1} Name", key=f"var_name_{i}", placeholder="text")
            with col2:
                var_value = st.text_input(f"Variable {i+1} Value", key=f"var_value_{i}", placeholder="Your value")
            if var_name:
                vars_dict[var_name] = var_value
        
        submitted = st.form_submit_button("➕ Add Prompt", width='stretch')
        
        if submitted:
            if prompt_id and prompt_template:
                new_prompt = {
                    "id": prompt_id,
                    "prompt": prompt_template,
                    "vars": vars_dict if vars_dict else {}
                }
                st.session_state.manual_prompts.append(new_prompt)
                st.success(f"✓ Added prompt: {prompt_id}")
                st.rerun()
            else:
                st.error("Please fill in Prompt ID and Prompt Template")
    
    # Display current prompts
    if st.session_state.manual_prompts:
        st.divider()
        st.subheader(f"Current Prompts ({len(st.session_state.manual_prompts)})")
        
        for i, prompt in enumerate(st.session_state.manual_prompts):
            with st.expander(f"Prompt {i+1}: {prompt.get('id', 'unnamed')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json(prompt)
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.manual_prompts.pop(i)
                        st.rerun()
        
        if st.button("✅ Use These Prompts", type="primary", width='stretch'):
            return st.session_state.manual_prompts.copy()
    
    return None


def _show_example_prompt_format():
    """Show example prompt format."""
    st.subheader("📝 Example Prompt Format")
    example = {
        "id": "summary_test",
        "prompt": "Summarize this article in 3 bullet points: {{text}}",
        "vars": {
            "text": "Large language models are changing software..."
        }
    }
    st.json(example)
    st.info("💡 You can type/paste JSON, upload a file, use examples, or add prompts manually using the form.")


if __name__ == "__main__":
    main()
