from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
from autogen_agentchat.agents import AssistantAgent 
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from logging_config import get_logger
import base64
import io

logger = get_logger(__name__)
load_dotenv()

# Initialize with empty DataFrame
df = pd.DataFrame()

# Only load default data if no user data is present
def initialize_default_data():
    global df
    if df.empty:
        df = pd.read_csv("mba_decision_dataset-mini.csv")
        df.columns = df.columns.str.strip()
    return df

# Initialize default data
initialize_default_data()

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT") or "",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT") or "",
    model="gpt-4o-2024-05-13",
    api_version="2024-02-01",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY") or "",
)

def get_dataframe_rows(dataframe):
    if len(dataframe) > 100:
        return dataframe.head(100).to_string(index=False)
    else:
        return dataframe.to_string(index=False)


# Update the Assistant initialization to be a function
def create_assistant(dataframe):
    csv_string = get_dataframe_rows(dataframe)
    return AssistantAgent(
        name="visualization_expert",
        model_client=model_client,
        tools=[],
        description="Agent to analyze CSV data and create visualizations",
        system_message=f"""You are a data visualization expert using plotly express.
        You have access to a DataFrame with these columns: {list(dataframe.columns)}.
        All rows of data: {csv_string}
        Return ONLY the Python code to create the visualization.
        Do not include fig.show() in your code as this is a Dash application.
        Ensure your response is a JSON object with this structure:
        {{
            "code_response_list": [
                {{
                    "code": "import plotly.express as px\\nfig = px.scatter(...) or other plotly express function",
                    "code_type": "visualization",
                    "observation": ""**Must include:** 
                        - Key patterns/trends 
                        - Notable anomalies 
                        - Statistical significance"
                }}
            ]
        }}
        
        **Visualization Guidelines:**
        . **Data Analysis:**
        - **Before visualization, perform thorough analysis of specified columns:**
            * Identify trends (temporal patterns, progression)
            * Detect anomalies (unexpected values/spikes/drops)
            * **Find correlations between target columns**
            * **Cluster similar data points**
        - **Incorporate findings into visualization design**
        
         **Chart Selection if the USer hasnt specifically stated a chart type:**
        - Use appropriate Plotly Express functions (px.line, px.scatter, px.bar, px.box, px.histogram, px.imshow, px.treemap)
        - Match chart type to data relationships:
            * Trends over time → px.line
            * Correlations → px.scatter or px.imshow
            * Distributions → px.histogram/px.box
            * Hierarchical data → px.treemap/px.sunburst
            * Multivariate → px.parallel_coordinates

        NOTE:
        - Use px.line, px.scatter, px.bar, or other appropriate plotly express charts, and px.box for distribution.
            - The generated code should be executable and free from errors, Use pd.concat() to combine multiple dataframes.
            - Provide a brief explanation of observed trend/pattern/insight/summary of the data.
        Example
        Use Case:
        px.line_mapbox → trends over time (eg. salary distribution by city)
            1. Heatmaps for Correlation Analysis
        Function: px.imshow(df.corr(), text_auto=True, color_continuous_scale='Viridis')
        Use Case: Visualizing correlation matrices between numerical columns.
        2. Histograms for Data Distribution
        Function: px.histogram(df, x='Salary', nbins=20, title='Salary Distribution')
        Use Case: Understanding distribution of numerical variables.
        3. Treemaps for Hierarchical Data
        Function: px.treemap(df, path=['Industry', 'Job Role'], values='Salary')
        Use Case: Displaying nested categorical data like industry vs. salary.
        4. Bubble Charts for Weighted Scatter Plots
        Function: px.scatter(df, x='GPA', y='Salary', size='Work Experience', color='Industry')
        Use Case: Showing relationships with an extra size dimension.
        5. Sunburst Charts for Multi-Level Categories
        Function: px.sunburst(df, path=['Country', 'State', 'City'], values='Population')
        Use Case: Drill-down visualization for regional or hierarchical data
        """
    )

# Initialize Assistant with default data
Assistant = create_assistant(df)

def get_fig_from_code(code_response):
    try:
        import json
        # Handle both string and dictionary responses
        if isinstance(code_response, str):
            # Clean the response string
            cleaned_response = code_response.replace('```json', '').replace('```', '').strip()
            response_dict = json.loads(cleaned_response)
        else:
            response_dict = code_response

        # Get the first visualization code
        if response_dict.get('code_response_list'):
            code = response_dict['code_response_list'][0]['code']
            
            # Remove any show() calls and clean the code
            code = code.replace('fig.show()', '').strip()
            
            # Create a safe local environment for execution
            local_vars = {
                "df": df, 
                "px": px,
                "pd": pd,
                "fig": None
            }
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            if 'fig' not in local_vars or local_vars['fig'] is None:
                logger.error("No figure object created in the code")
                return None
                
            return {
                'figure': local_vars['fig'],
                'code': code,
                'observation': response_dict['code_response_list'][0].get('observation', 'No observation provided')
            }
    except json.JSONDecodeError as je:
        logger.error(f"JSON parsing error: {str(je)}")
        return None
    except Exception as e:
        logger.error(f"Error in visualization generation: {str(e)}")
        return None

async def create_visualization(user_input):
    logger.info(f"Creating visualization for user input: {user_input}")
    task_result = await Assistant.run(task=user_input)
    logger.info(f"Visualization created result: {task_result.messages[-1].content}")
    return task_result.messages[-1].content

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Data Visualization", style={'textAlign': 'center', 'margin': '20px'}),
    
    # Add Upload component
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a CSV File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0'
            },
            multiple=False
        ),
        html.Div(id='upload-status'),
    ], style={'width': '90%', 'margin': '0 auto'}),
    
    # Existing components
    html.Div([
        dcc.Textarea(
            id="user_input",
            placeholder="Enter your visualization request...",
            style={
                "width": "100%",
                "height": 100,
                "margin": "10px 0",
                "padding": "10px",
                "borderRadius": "5px"
            }
        ),
        html.Button(
            "Generate Plot",
            id="generate-button",
            style={
                "backgroundColor": "#119DFF",
                "color": "white",
                "border": "none",
                "padding": "10px 20px",
                "borderRadius": "5px",
                "cursor": "pointer",
                "margin": "10px 0"
            }
        ),
    ], style={'width': '90%', 'margin': '0 auto'}),
    
    dcc.Loading(
        children=[
            html.Div(id="my-figure"),
            html.Div(id="content", style={'margin': '20px'})
        ],
        type="cube",
        color="#119DFF"
    ),
    dag.AgGrid(
        id='AgGrid',  # Add an ID to the AgGrid
        rowData=df.to_dict("records"),
        columnDefs=[{"field": i} for i in df.columns],
        defaultColDef={"filter": True, "sortable": True, "floatingFilter": True},
        style={'height': '400px', 'margin': '20px 0'}
    ),
], style={'padding': '20px'})

@callback(
    Output("my-figure", "children"),
    Output("content", "children"),
    Input("generate-button", "n_clicks"),
    State("user_input", "value"),
    prevent_initial_call=True
)
def create_graph(n_clicks, user_input):
    if not user_input:
        return "", "Please enter a visualization request."
    
    try:
        import asyncio
        response = asyncio.run(create_visualization(user_input))
        
        result = get_fig_from_code(response)
        if not result:
            return html.Div([
                html.P("Error: Could not generate visualization", style={'color': 'red'}),
                html.Pre(str(response), style={'background': '#f8f9fa', 'padding': '10px'})
            ]), "Failed to process the visualization code."

        figure = result['figure']
        code = result['code']
        observation = result['observation']
        
        return dcc.Graph(
            figure=figure,
            style={'height': '600px', 'width': '100%'}
        ), html.Div([
            html.H4("Generated Code:"),
            html.Pre(code, style={
                'background': '#f8f9fa',
                'padding': '15px',
                'border-radius': '5px',
                'margin': '10px 0'
            }),
            html.H4("Observation:"),
            html.P(observation, style={'font-style': 'italic'})
        ])
            
    except Exception as e:
        logger.error(f"Error in create_graph: {str(e)}")
        return html.Div([
            html.P(f"Error: {str(e)}", style={'color': 'red'})
        ]), ""

# Add new callback for CSV upload
@callback(
    Output('upload-status', 'children'),
    Output('AgGrid', 'rowData'),
    Output('AgGrid', 'columnDefs'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_data(contents):
    global df, Assistant  # Move global declaration to the start of the function
    
    if contents is None:
        return "No file uploaded.", df.to_dict('records'), [{"field": i} for i in df.columns]
    
    new_df = parse_csv_contents(contents)
    
    if new_df is not None:
        df = new_df  # Update the global df
        Assistant = create_assistant(df)  # Create new Assistant with updated data
        
        return (
            html.Div([
                html.P("File uploaded successfully!", style={'color': 'green'}),
                html.P(f"Columns: {', '.join(df.columns)}")
            ]),
            df.to_dict('records'),
            [{"field": i} for i in df.columns]
        )
    
    return html.P("Error processing file!", style={'color': 'red'}), [], []

def parse_csv_contents(contents):
    """Parse uploaded CSV file contents"""
    if contents is None:
        return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        return None

if __name__ == "__main__":
    app.run_server(debug=True) 