from openai import OpenAI, AzureOpenAI
import streamlit as st
import plotly.graph_objects as go
import json
import re
import csv
import io
import os
from dotenv import load_dotenv

class LLMService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Try to get OpenAI API key from st.secrets first, then fallback to .env
        api_key = None
        
        # First try st.secrets (for Streamlit Cloud deployment)
        try:
            if 'AZURE_OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['AZURE_OPENAI_API_KEY']
            if 'ENDPOINT_URL' in st.secrets:
                endpoint = st.secrets['ENDPOINT_URL']
        except Exception:
            # st.secrets might not be available in all contexts
            pass
        
        # If not found in st.secrets, try .env file
        if not api_key or not endpoint:
            endpoint = os.getenv("ENDPOINT_URL")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        # If still not found, raise an error
        if not api_key or not endpoint:
            raise ValueError(
                'Azure OpenAI API or Endpoint URL not found. Please add it to either:' + chr(10) +
                '1. Streamlit secrets (for deployment): Add to your secrets.toml file' + chr(10) +
                '2. Environment file: Add AZURE_OPENAI_API_KEY and ENDPOINT_URL to your .env file'
            )

        # Initialize Azure OpenAI client with key-based authentication
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )

        
    def get_evaluation(self, prompt):
        """Get evaluation results, return raw JSON data"""
        try:
            chat_prompt = [
                {
                    "role": "system", "content": [
                        {
                            "type": "text",
                            "text": '''You are a professional systemic investing evaluation expert. Your task is to assess investment cases based on the 13 hallmarks framework.

    IMPORTANT: You MUST return your response in the following JSON format:
    {
        "table": "A markdown formatted table with columns: Hallmark, Score (0-10), Justification, Suggested Indicators",
        "overall_score": "The average score as a number with one decimal point",
        "scores": {
            "Systems Thinking and Complexity Science": score1,
            "Paradigm Evolution": score2,
            ...
        }
    }

    For each hallmark, you should:
    1. Provide a score from 0 to 10 (with one decimal point)
    2. Give a brief justification for the rating
    3. Suggest relevant indicators from the provided indicator set

    The table should be formatted in markdown with clear column headers.
    The overall_score should be a number with one decimal point.
    The scores object should use the full Hallmark title as the key, matching the Hallmark column in the table, and contain all 13 hallmark scores as numbers.

    Ensure your evaluation is thorough, objective, and well-justified.'''}
                    ]
                },
                {
                    "role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Include speech result if speech is enabled
            messages = chat_prompt

            # Generate the completion
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stop=None,
                stream=False,
                temperature=0,
                seed=42
            )

            # Parse JSON response
            response_text = completion.choices[0].message.content
            try:
                # Try to parse JSON directly
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract and clean JSON part
                try:
                    # Find JSON start and end positions
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = response_text[start:end]
                        # Clean JSON string
                        cleaned_json = self.clean_json_string(json_str)
                        result = json.loads(cleaned_json)
                    else:
                        raise json.JSONDecodeError("No JSON object found", response_text, 0)
                except Exception as e:
                    raise ValueError(f"Failed to parse the response as JSON: {str(e)}")
            # Ensure scores is dict type
            if 'scores' in result and isinstance(result['scores'], str):
                try:
                    result['scores'] = json.loads(result['scores'])
                except Exception:
                    raise ValueError("The scores field returned by LLM is a string and cannot be parsed as a dictionary.")
            if 'scores' in result and not isinstance(result['scores'], dict):
                raise ValueError("The scores field returned by LLM is not a dictionary.")
            return result
                
        except Exception as e:
            raise Exception(f"Error getting LLM response: {str(e)}")

    def clean_json_string(self, json_str):
        """Clean JSON string"""
        # Remove control characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
        # Handle line breaks
        json_str = json_str.replace('\n', '\\n')
        return json_str

class EvaluationVisualizer:
    def __init__(self):
        # Load mapping files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        level_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_files', 'system_change_level_to_hallmarks.json')
        condition_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_files', 'system_change_condition_to_hallmarks.json')
        with open(level_path, 'r', encoding='utf-8') as f:
            self.level_map = json.load(f)
        with open(condition_path, 'r', encoding='utf-8') as f:
            self.condition_map = json.load(f)

    def create_radar_chart(self, scores: dict, height=1000, width=1000) -> go.Figure:
        # Create radar chart directly using scores dictionary
        categories = list(scores.keys())
        values = list(scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Hallmark Scores'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
                angularaxis=dict(tickangle=0, tickfont=dict(size=12))
            ),
            showlegend=False,
            title=dict(text="Hallmark Scores Radar Chart", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            height=height,
            width=width,
            margin=dict(l=150, r=150, t=100, b=100)
        )
        return fig

    def create_level_radar_chart(self, scores: dict, height=800, width=800) -> go.Figure:
        categories = []
        values = []
        for level, hallmarks in self.level_map.items():
            hallmark_scores = [scores.get(h, None) for h in hallmarks if h in scores]
            hallmark_scores = [s for s in hallmark_scores if s is not None]
            avg = round(sum(hallmark_scores)/len(hallmark_scores), 2) if hallmark_scores else 0
            categories.append(level)
            values.append(avg)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='System Change Level Scores'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
                angularaxis=dict(tickangle=0, tickfont=dict(size=12))
            ),
            showlegend=False,
            title=dict(text="System Change Level Radar Chart", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            height=height,
            width=width,
            margin=dict(l=160, r=160, t=80, b=80)
        )
        return fig

    def create_condition_radar_chart(self, scores: dict, height=800, width=800) -> go.Figure:
        categories = []
        values = []
        for cond, hallmarks in self.condition_map.items():
            hallmark_scores = [scores.get(h, None) for h in hallmarks if h in scores]
            hallmark_scores = [s for s in hallmark_scores if s is not None]
            avg = round(sum(hallmark_scores)/len(hallmark_scores), 2) if hallmark_scores else 0
            categories.append(cond)
            values.append(avg)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='System Change Condition Scores'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
                angularaxis=dict(tickangle=0, tickfont=dict(size=12))
            ),
            showlegend=False,
            title=dict(text="System Change Condition Radar Chart", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            height=height,
            width=width,
            margin=dict(l=100, r=100, t=80, b=80)
        )
        return fig

    def create_merged_level_condition_radar(self, scores: dict) -> go.Figure:
        # Calculate average score for level and condition
        level_categories = list(self.level_map.keys())
        level_values = []
        for level, hallmarks in self.level_map.items():
            hallmark_scores = [scores.get(h, None) for h in hallmarks if h in scores]
            hallmark_scores = [s for s in hallmark_scores if s is not None]
            avg = round(sum(hallmark_scores)/len(hallmark_scores), 2) if hallmark_scores else 0
            level_values.append(avg)
        condition_categories = list(self.condition_map.keys())
        condition_values = []
        for cond, hallmarks in self.condition_map.items():
            hallmark_scores = [scores.get(h, None) for h in hallmarks if h in scores]
            hallmark_scores = [s for s in hallmark_scores if s is not None]
            avg = round(sum(hallmark_scores)/len(hallmark_scores), 2) if hallmark_scores else 0
            condition_values.append(avg)
        # Merge all dimensions
        all_categories = level_categories + condition_categories
        level_plot = level_values + [None]*len(condition_categories)
        condition_plot = [None]*len(level_categories) + condition_values
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=level_plot,
            theta=all_categories,
            fill='toself',
            name='System Change Levels',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatterpolar(
            r=condition_plot,
            theta=all_categories,
            fill='toself',
            name='System Change Conditions',
            line=dict(color='orange')
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10]),
                angularaxis=dict(tickangle=0, tickfont=dict(size=12))
            ),
            showlegend=True,
            title=dict(text="System Change Levels & Conditions Radar Chart", y=0.95, x=0.5, xanchor='center', yanchor='top'),
            height=900,
            width=900,
            margin=dict(l=160, r=160, t=80, b=80)
        )
        return fig

    def display_evaluation(self, result):
        """Display evaluation results"""
        # Display table
        st.markdown(result['table'])
        
        # Display overall score
        st.subheader("Overall Score")
        st.write(f"Average Score: {result['overall_score']}")
        
        # Create and display radar chart
        st.subheader("Score Distribution")
        try:
            fig = self.create_radar_chart(result['scores'])
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating radar chart: {str(e)}")

        fig1 = self.create_level_radar_chart(result['scores'])
        fig2 = self.create_condition_radar_chart(result['scores'])
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        fig3 = self.create_merged_level_condition_radar(result['scores'])
        st.plotly_chart(fig3) 
