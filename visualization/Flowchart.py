# Databricks notebook source
# MAGIC %md
# MAGIC # Running Instruction:

# COMMAND ----------

# MAGIC %md
# MAGIC Just upload the notebook to google colab and run it. 
# MAGIC
# MAGIC This is supposed to be the general layout of the webpage that is displayed to the user in linkedin after he submits the query. We don't have any experience in web development so this took effort even though it's pretty basic, but it shows what we envisioned when we built the project together!

# COMMAND ----------

!pip install plotly
!pip install dash
!pip install dash-dangerously-set-inner-html
!pip install markdown

# COMMAND ----------

import plotly.graph_objs as go
import dash
import markdown
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
from dash import dcc, html, Input, Output

# COMMAND ----------

# MAGIC %md
# MAGIC # Parsing

# COMMAND ----------

def parse_response(text: str):
    """
    Parse the response text and return a list of nodes.

    Parameters
    ----------
    text : str, response text

    Returns
    -------
    nodes : list, nodes in the event flow chart

    """

    nodes = []

    html_text = markdown.markdown(text)
    step_num = 0
    for line in html_text.split("<p>"):
        if line == "":
            continue
        line = "<p>" + line
        title = (line.split("</p>")[0] + "</p>").strip("\n")
        info = line.split("</p>")[1].strip("\n")
        if title.lower().find("step") != -1:
            nodes.append({"Title": title, 'pos': (step_num, 0), "info": info})
            step_num += 1

    return nodes

# COMMAND ----------

# MAGIC %md
# MAGIC # Plotly Figure

# COMMAND ----------

def create_figure(nodes, size, flowchart_title):
    """
    Create a basic event flow chart figure using Plotly.

    Parameters
    ----------
    nodes : list, nodes in the event flow chart
    size : float, size of the squares

    Returns
    -------
    fig : plotly.graph_objs.Figure, basic event flow chart figure

    """
    # Create the figure
    fig = go.FigureWidget()
    fig.update_layout(
        autosize=False,
        width=200 * len(nodes),  # Adjust based on your preference
        height=300,  # Adjust to maintain aspect ratio
    )

    # Offset for arrow start/end points to prevent crossing the square borders
    arrow_offset = size * 2  # Adjust this factor based on the size of the square and desired gap

    # Add square shapes for each node
    for i, details in enumerate(nodes):
        fig.add_shape(type="rect",
                    x0=details['pos'][0]-size, y0=details['pos'][1]-size,
                    x1=details['pos'][0]+size, y1=details['pos'][1]+size,
                    line=dict(color="Black"),
                    fillcolor="LightSkyBlue")

        # Add text annotations in the center of squares
        fig.add_annotation(
            x=details['pos'][0],
            y=details['pos'][1],
            text=f"Step {i + 1}",  # Node text
            showarrow=False,
            font=dict(size=10))  # Adjust font size as needed

    # Add paths (as arrows)
    for i in range(len(nodes) - 1):
        # Get start and end positions
        start_pos = nodes[i]['pos']
        end_pos = nodes[i + 1]['pos']

        # Calculate adjusted positions for arrow start (ax, ay) and end (x, y)
        ax_adjusted = start_pos[0] + arrow_offset - size  # Start a bit right of the left square border
        ay_adjusted = start_pos[1]  # Keep y the same
        x_adjusted = end_pos[0] - arrow_offset + size  # End a bit left of the right square border
        y_adjusted = end_pos[1]  # Keep y the same

        # Add annotation (arrow) with adjusted positions
        fig.add_annotation(
            x=x_adjusted,  # Adjusted arrowhead position
            y=y_adjusted,  # Adjusted arrowhead position
            ax=ax_adjusted,  # Adjusted arrow tail position
            ay=ay_adjusted,  # Adjusted arrow tail position
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # No text inside arrow
            showarrow=True,
            arrowhead=2,  # Arrowhead style
            arrowsize=1,  # Arrowhead size
            arrowwidth=2,  # Arrowhead width
            arrowcolor='black'  # Arrowhead color
        )

    # Update layout
    fig.update_layout(title=flowchart_title,
                    showlegend=False)
    fig.update_xaxes(range=[0 - size * 2, (len(nodes) - 1) + size * 2],
                     showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-size * 2, size * 2],
                     showgrid=False, zeroline=False, showticklabels=False)
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC # Dash Interactive App

# COMMAND ----------

def create_div(index, node, fig) -> dash.html.Div:

    horizontal_padding = 80  # Adjust this value to reduce spread to the sides
    vertical_padding = 5  # Adjust this value to raise the overlay higher

    overlay_left = ((node['pos'][0] - fig.layout.xaxis.range[0]) / (fig.layout.xaxis.range[1] - fig.layout.xaxis.range[0]))
    overlay_left = overlay_left * (fig.layout.width - 2 * horizontal_padding) + horizontal_padding

    overlay_top = ((node['pos'][1] - fig.layout.yaxis.range[0]) / (fig.layout.yaxis.range[1] - fig.layout.yaxis.range[0]))
    overlay_top = overlay_top * (fig.layout.height - 2 * vertical_padding) + vertical_padding

    # Invert y-axis for HTML positioning (CSS top increases downwards)
    overlay_top = fig.layout.height - overlay_top + (2 * vertical_padding)

    # Convert shape dimensions to pixels
    data_width_units = fig.layout.xaxis.range[1] - fig.layout.xaxis.range[0]  # Width of the data in data units
    data_height_units = fig.layout.yaxis.range[1] - fig.layout.yaxis.range[0]  # Height of the data in data units

    shape_width_units = fig.layout['shapes'][index]['x1'] - fig.layout['shapes'][index]['x0']
    shape_height_units = fig.layout['shapes'][index]['y1'] - fig.layout['shapes'][index]['y0']

    shape_width_pixels = (shape_width_units / data_width_units) * fig.layout.width - 25
    shape_height_pixels = (shape_height_units / data_height_units) * fig.layout.height - 100

    div = html.Div(
        '',
        id=f'node-{index}',
        style={
            'width': f'{shape_width_pixels}px',
            'height': f'{shape_height_pixels}px',
            'position': 'absolute',
            'top': f'{overlay_top}px',
            'left': f'{overlay_left}px',
            'background-color': 'rgba(0, 0, 0, 0)',
            'padding': '5px',
            'cursor': 'pointer',
            'transform': 'translate(-50%, -50%)'  # Center the div on the node
        }
    )

    return div

# COMMAND ----------

def visualize_flowchart(response_text):
    # Nodes of the event flow chart with additional info
    nodes = parse_response(text=response_text)

    flowchart_title = f"Career Path for a LinkedIn User in the Field of Data"

    # Create the figure and setup the callback function
    fig = create_figure(nodes=nodes, size=0.3, flowchart_title=flowchart_title)

    app = dash.Dash(prevent_initial_callbacks="initial_duplicate")

    flowchart = dash.dcc.Graph(
            id='flowchart-graph',
            figure=fig,
            style={'width': '100%', 'height': '100%'}
        )

    additional_data_div = html.Div(
            '',
            id=f'additional-data',
            style={
                'position': 'absolute',
                'background-color': 'rgba(255, 255, 255, 1)',
                'padding': '15px'
            }
        )

    app.layout = html.Div(style={'position': 'relative', 'width': '500px', 'height': '300px', 'cursor': 'default'},
    children=[flowchart,*[create_div(index=i, node=node, fig=fig) for i, node in enumerate(nodes)], additional_data_div])

    @app.callback(
        Output('additional-data', 'children'),
        [Input(f'node-{i}', 'n_clicks') for i, _ in enumerate(nodes)]
    )
    def update_additional_data(*args):
        ctx = dash.callback_context
        if not ctx.triggered:
            # No clicks yet
            return "Click on a step to read its details!"
        else:
            # Use callback_context to determine which node was clicked
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            node_index = int(triggered_id.split('-')[-1])
            return DangerouslySetInnerHTML(nodes[node_index]['Title'] + "<br>" + nodes[node_index]['info'])

    app.run_server(debug=True, use_reloader=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizations of Example LLM Responses

# COMMAND ----------

# MAGIC %md
# MAGIC Good Example:

# COMMAND ----------

response_text = """**Career Path to Become a Machine Learning Engineer at Google, Amazon, or Meta**

**Step 1: Obtain a Strong Technical Foundation**

* Pursue a bachelor's or master's degree in computer science, engineering, data science, or a related technical field from a reputable university.
* Focus on developing a solid understanding of data structures, algorithms, software development, and machine learning principles.
* Consider obtaining specialized certifications in cloud computing platforms (e.g., AWS, Google Cloud Platform) to enhance your technical credibility.

**Step 2: Gain Experience in Data-Related Roles**

* Seek internships or entry-level positions as data scientists, data engineers, or data analysts to gain hands-on experience in data analysis and interpretation.
* Work on projects that involve collecting, cleaning, and analyzing large datasets using tools such as Hadoop and Spark.
* Develop proficiency in programming languages like Python and R, which are widely used in data science and machine learning.

**Step 3: Develop Machine Learning Expertise**

* Take online courses or attend workshops to enhance your knowledge of machine learning algorithms, model evaluation, and optimization techniques.
* Build a portfolio of personal projects that demonstrate your ability to apply machine learning to solve real-world problems.
* Consider pursuing a graduate degree in machine learning or a related field to deepen your understanding and gain specialized knowledge.

**Step 4: Acquire Engineering and Development Skills**

* Develop strong software engineering skills by working on projects that involve designing, implementing, and testing software solutions.
* Gain experience in cloud computing platforms and their applications in various industries.
* Learn about agile development methodologies and best practices for software engineering.

**Step 5: Build a Network and Seek Mentorship**

* Attend industry events and conferences to connect with professionals in the field of machine learning and data science.
* Seek mentorship from experienced machine learning engineers at your company or through professional organizations.
* Join online communities and forums to engage with other professionals and stay updated on industry trends.

**Step 6: Apply for Machine Learning Engineering Roles**

* Target companies like Google, Amazon, and Meta that are known for their focus on machine learning and artificial intelligence.
* Tailor your resume and cover letter to highlight your technical skills, experience in data-related roles, and passion for machine learning.
* Prepare for technical interviews by practicing coding challenges and reviewing machine learning concepts.

**Step 7: Continuous Learning and Development**

* Stay abreast of the latest advancements in machine learning and data science by attending conferences, reading research papers, and experimenting with new technologies.
* Seek opportunities for professional development through online courses, workshops, or certifications.
* Engage in side projects or contribute to open-source projects to showcase your skills and stay competitive in the job market."""

# COMMAND ----------

# MAGIC %md
# MAGIC Bad Example:

# COMMAND ----------

response_text = """**Career Path for a Healthcare Professional Aspiring to Become a Lumberjack in Construction and Software Development**

**Step 1: Obtain a Bachelor's Degree in Construction or Software Development**

* Consider pursuing a degree in construction management, software engineering, or a related field.
* Explore programs that offer a balance of theoretical knowledge and practical experience.

**Step 2: Gain Experience in Construction Management**

* Seek internships or entry-level positions in construction companies.
* Focus on developing skills in project management, safety, and construction techniques.

**Step 3: Develop Software Development Skills**

* Enroll in online courses or bootcamps to acquire proficiency in software design, development, and testing.
* Build a portfolio of personal projects to demonstrate your abilities.

**Step 4: Obtain a Master's Degree in Healthcare Administration**

* Consider pursuing a master's degree in healthcare administration to enhance your understanding of the healthcare industry.
* This will provide you with a competitive edge in the construction and software development sectors.

**Step 5: Seek Leadership Roles**

* Take on leadership responsibilities within your current or future roles.
* Demonstrate your ability to manage projects, motivate teams, and drive results.

**Step 6: Network with Professionals in Construction and Software Development**

* Attend industry events and conferences to connect with potential employers.
* Join professional organizations and LinkedIn groups to expand your network.

**Step 7: Apply for Lumberjack Positions in Construction and Software Development Companies**

* Target companies that specialize in construction and software development.
* Highlight your unique combination of skills and experience in your resume and cover letter.
* Be prepared to demonstrate your passion for both industries and your ability to contribute to their success."""

# COMMAND ----------

visualize_flowchart(response_text=response_text)
