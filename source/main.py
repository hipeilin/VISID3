import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_bio
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
from datetime import timedelta, datetime
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from functions import *
from toggle_style import *
from variables import *

# pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------------
# Declaration
# --------------------------------------------------------------------------
interaction_dir = 'csv'

num_of_participant = 14
df_interaction_list, df_bubble_list, df_vc_list = [], [], []
df_year_list = []
for n in range(1, 1 + num_of_participant):
    _df_interaction = fetch_csv(interaction_dir + '/P' + str(n) + '_interaction.csv')
    _df_interaction = _df_interaction.replace(np.NaN, '')
    df_interaction_list.append(_df_interaction)

    _df_bubble = fetch_csv(interaction_dir + '/P' + str(n) + '_bubble.csv')
    _df_bubble = _df_bubble.replace(np.NaN, '')
    df_bubble_list.append(_df_bubble)

    _df_vc = fetch_csv(interaction_dir + '/P' + str(n) + '_vc.csv')  # Use '_vc' or '_bubble'
    _df_vc = _df_vc.replace(np.NaN, '')
    df_vc_list.append(_df_vc)

min_height = 20
height_scale = 50
comp_plot_height = min_height + num_of_participant * height_scale

task_score_df = pd.DataFrame()
task_scores_df = pd.DataFrame()

# --------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[

        ###########################################################################################
        # INDIVIDUAL VIEW
        ###########################################################################################
        html.Div(
            children=[
                #########
                # 1.0
                # Individual view heading
                html.H5('Individual View Panel', style={'background-color': 'darkGray', 'color': 'white',
                                                        'padding': '5px', 'margin': '0px', 'font-weight': 'bold'}),

                #########
                # 1.1
                # Select participant
                html.Div([
                    html.P('Select participant', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='participant_selection',
                        multi=False,
                        clearable=False,
                        options=[
                            {'label': 'P' + str(i), 'value': i} for i in range(1, 1 + num_of_participant)
                        ],
                        value=1,
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'width': '14vw', 'verticalAlign': 'top', 'padding': '5px'}),

                #########
                # 1.2
                # Select task
                html.Div([
                    # Task selection
                    html.P('Task selection', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='task_selection_individual',
                        options=[{'label': 'All', 'value': 'All'},
                                 {'label': 'Task 1', 'value': 'Task 1'},
                                 {'label': 'Task 2', 'value': 'Task 2'},
                                 {'label': 'Task 3', 'value': 'Task 3'},
                                 {'label': 'Task 4', 'value': 'Task 4'},
                                 {'label': 'Task 5', 'value': 'Task 5'}],
                        value='All',
                    ),
                ], style={'display': 'inline-block', 'width': '11vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 1.3
                # Toggle between plotting interactions or complete bubbles
                html.Div([
                    html.P('Granular level', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='bubble_length_switch',
                        options=[{'label': 'Interaction (low)', 'value': 'next'},
                                 {'label': 'Holistic (high)', 'value': 'life'}],
                        value='next',
                    ),
                    html.P('Infopanel color by', style={'margin': '20px 0px 0px 0px', 'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='bubble_color_switch',
                        options=[{'label': 'Unified', 'value': 'unified'},
                                 {'label': 'Municipality', 'value': 'municipality'},
                                 ],
                        value='unified',
                    ),
                ], style={'display': 'inline-block', 'width': '13vw',
                          'padding': '5px', 'verticalAlign': 'top'}),

                #########
                # 1.4
                # Add attribute to the individual plot
                html.Div(
                    children=[
                        html.P('Attribute selection', style={'margin': '0px', 'font-weight': 'bold'}),
                        dcc.Checklist(
                            id='indi_attribute_selection',
                            options=[{'label': 'Add "year" data', 'value': 'show_year_individual'},
                                     {'label': 'Add "gender" data', 'value': 'show_gender_individual'},
                                     {'label': 'Add "municipality" data', 'value': 'show_municipality_individual'}
                                     ],
                            value=[],
                            style={'display': 'inline-block', 'verticalAlign': 'top'},
                        ),
                    ], style={'display': 'inline-block', 'width': '13vw',
                              'padding': '5px', 'verticalAlign': 'top'}),

                #########
                # 1.5
                # Filter & reset click selection
                html.Div([
                    html.Span('Filter events duration (s) shorter than',
                              style={'margin': '0px', 'font-weight': 'bold', 'display': 'block'}),
                    dcc.Input(id='filter_bubble_duration_indi', type='number', placeholder='0',
                              debounce=True,
                              style={'width': '60px', 'marginRight': '5px'}),
                    html.Button('Filter', id='filter_bubble_indi',
                                style={'width': '120px', 'marginRight': '5px'}),
                    html.Button('Reset', id='reset_filter_duration_indi',
                                style={'width': '100px'}),

                ], style={'display': 'inline-block', 'width': '19vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 1.6
                # Main plot
                dcc.Graph(id='timeline_plot', config={'displayModeBar': False},
                          style={'padding': '5px', 'verticalAlign': 'top', 'width': '97vw'}),

            ],
            style={'border': '1px darkGray solid', 'padding': '0px', 'margin': '0px',
                   'box-shadow': '6px 6px 6px 1px lightGray'}
        ),

        html.Br(),
        ###########################################################################################
        # COMPARISON VIEW
        ###########################################################################################

        html.Div(
            children=[
                #########
                # 2.0
                # Comparison view heading
                html.H5('Comparison View Panel', style={'background-color': 'darkGray', 'color': 'white',
                                                        'padding': '5px', 'margin': '0px', 'font-weight': 'bold'}),
                #########
                # 2.1
                # Select participant
                html.Div([
                    html.P('Select participant as baseline', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='participant_anchor_comp',
                        multi=False,
                        clearable=False,
                        options=[
                            {'label': 'P' + str(i), 'value': i} for i in range(1, 1 + num_of_participant)
                        ],
                        value=1,
                        style={'width': '200px'}
                    ),
                ], style={'display': 'inline-block', 'width': '14vw', 'verticalAlign': 'top', 'padding': '5px'}),

                #########
                # 2.2
                # Task selection
                html.Div([
                    html.P('Task selection', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='task_selection_comp',
                        options=[{'label': 'Task 1', 'value': 'Task 1'},
                                 {'label': 'Task 2', 'value': 'Task 2'},
                                 {'label': 'Task 3', 'value': 'Task 3'},
                                 {'label': 'Task 4', 'value': 'Task 4'},
                                 {'label': 'Task 5', 'value': 'Task 5'}, ],
                        value='Task 1',
                        style={'marginBottom': '3px'}
                    ),
                    dcc.Checklist(
                        id='task_aligned',
                        options=[{'label': 'Align by task', 'value': True}],
                        value=[],
                    ),
                ], style={'display': 'inline-block', 'width': '11vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 2.3
                # Comparison plot customization
                html.Div([
                    html.P('Background color by', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.RadioItems(
                        id='color_switch_comp',
                        options=[{'label': 'View', 'value': 'view'},
                                 {'label': 'Municipality', 'value': 'municipality'},
                                 {'label': 'None', 'value': 'none'}],
                        value='view',
                    ),
                ], style={'display': 'inline-block', 'width': '13vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 2.4
                # Attribute selection
                html.Div([
                    html.P('Attribute selection', style={'margin': '0px', 'font-weight': 'bold'}),
                    dcc.Checklist(
                        id='attribute_selection_comp',
                        options=[{'label': 'Add "year" data', 'value': 'show_year_comp'},
                                 {'label': 'Add "gender" data', 'value': 'show_gender_comp'},
                                 ],
                        value=['show_year_comp', 'show_gender_comp'],
                    ),
                ], style={'display': 'inline-block', 'width': '13vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 2.5
                # Filter & reset click selection
                html.Div([
                    html.Span('Filter events duration (s) shorter than',
                              style={'margin': '0px', 'font-weight': 'bold', 'display': 'block'}),
                    dcc.Input(id='filter_bubble_duration_comparison', type='number', placeholder='0',
                              debounce=True,
                              style={'width': '60px', 'marginRight': '5px'}),
                    html.Button('Filter', id='filter_bubble_comparison',
                                style={'width': '120px', 'marginRight': '5px'}),
                    html.Button('Reset', id='reset_filter_duration_comparison',
                                style={'width': '100px'}),

                    html.Span('Click on background bars to focus',
                              style={'marginTop': '40px', 'font-weight': 'bold', 'display': 'block'}),
                    html.Button('Reset selection', id='reset_selection',
                                style={'width': '280px'}),

                ], style={'display': 'inline-block', 'width': '21vw', 'padding': '5px',
                          'verticalAlign': 'top'}),

                #########
                # 2.6
                # Heatmap ranking
                html.Div([
                    dcc.Checklist(
                        id='heatmap_rank',
                        options=[{'label': 'Show cluster', 'value': 'heatmap_ranked'}],
                        value=[],
                    ),

                    dcc.Graph(id='similarity_heatmap', config={'displayModeBar': False},
                              style={'display': 'inline-block'}),
                ], style={'display': 'inline-block', 'width': '18vw', 'padding': '5px', 'margin-left': '5vw',
                          'verticalAlign': 'top'}),

                #########
                # 2.7
                # Comparison plot
                dcc.Graph(id='comparison_plot', config={'displayModeBar': False},
                          style={'width': '80vw', 'display': 'inline-block'}),

                dcc.Graph(id='similarity_gauge', config={'displayModeBar': False},
                          style={'width': '12vw', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ],
            style={'border': '1px darkGray solid', 'padding': '0px', 'margin': '0px',
                   'box-shadow': '6px 6px 6px 1px lightGray'}
        ),

        # ================================================================================================
        # dummy element
        html.P(id='dummy', style={'display': 'none'}),
        html.Div(id='reorder_trigger'),
    ]
)


# Similarity gauge in Comparison view
@app.callback(
    Output('similarity_gauge', 'figure'),
    Input('reorder_trigger', 'children'),
)
def similarity_rank_plot_control(dummy):
    global task_score_df

    fig_gauge = make_subplots(rows=num_of_participant, cols=2,
                              specs=[[{'type': 'domain'}] * 2] * num_of_participant)

    score_list = task_score_df['score'].tolist()
    reordered_score_list = reorder_list_by_indexes(score_list, similarity_index_list)
    reordered_score_list = normalize_list(reordered_score_list)

    for p in range(0, num_of_participant):
        fig_gauge.add_trace(go.Indicator(
            # delta={'reference': 0},
            value=reordered_score_list[p],
            delta={'reference': max(reordered_score_list)},
            mode="number+gauge",
            gauge={
                'shape': "bullet",
                'axis': {'visible': False, 'range': [0, max(reordered_score_list)]},
                'bar': {'color': 'rgba(0, 95, 96,1)'},
            },
            number={'font_size': 10},
            domain={'row': 0, 'column': 0}),
            row=p + 1, col=1)

        # Add the indicator to show the difference in score
        fig_gauge.add_trace(go.Indicator(
            mode='delta',
            value=reordered_score_list[p],
            delta={'reference': max(reordered_score_list), 'font': {'size': 12}}),
            row=p + 1, col=2)

    fig_gauge.update_layout(
        bargap=0, autosize=False,
        margin={'l': 0, 'r': 10, 't': 10, 'b': 5},
        height=int(comp_plot_height),
        showlegend=False
    )

    return fig_gauge


# Heatmap in Similarity View
@app.callback(
    Output('similarity_heatmap', 'figure'),
    Input('reorder_trigger', 'children'),
    Input('heatmap_rank', 'value'),
)
def similarity_heatmap_control(dummy, heatmap_rank_checked):
    global task_scores_df

    color_map = 'YlGn'
    similarity_heat_df = pd.DataFrame(columns=[str(i) for i in range(1, num_of_participant + 1)])

    for i in task_scores_df:
        similarity_heat_df.loc[len(similarity_heat_df)] = i

    similarity_heat_df.index = similarity_heat_df.index + 1

    # ranked_heatmap_index_list = [x + 1 for x in similarity_index_list]  # Discarded

    if heatmap_rank_checked:
        # Use the heatmap with dendrogram
        fig_heatmap = dash_bio.Clustergram(
            data=similarity_heat_df,
            link_method='ward',
            column_labels=similarity_heat_df.columns.to_list(),
            row_labels=list(similarity_heat_df.index),
            width=530, height=280,
            center_values=False,  # Disable z-value and show the similarity
            color_map=color_map
        )
        fig_heatmap.update_layout(
            coloraxis_colorbar_x=-1
        )
    else:
        # Use the normal heatmap
        fig_heatmap = px.imshow(similarity_heat_df, y=similarity_heat_df.index,
                                color_continuous_scale=color_map,
                                labels=dict(x="", y="", color="Similarity"),
                                aspect='square', text_auto=True,
                                width=300, height=280,
                                )
        fig_heatmap.update_yaxes(autorange="reversed", type='category', side="right")
        fig_heatmap.update_xaxes(side="bottom", type='category')

    fig_heatmap.update_layout(
        bargap=1, autosize=False,
        margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
        yaxis=dict(scaleanchor='x'),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig_heatmap.update_layout(coloraxis_showscale=False)

    return fig_heatmap


###########################################################################################
# Comparison view controls
###########################################################################################

@app.callback(
    Output('comparison_plot', 'figure'),
    Output('reorder_trigger', 'children'),
    Input('task_selection_comp', 'value'),
    Input('attribute_selection_comp', 'value'),
    Input('color_switch_comp', 'value'),
    Input('task_aligned', 'value'),
    Input('comparison_plot', 'clickData'),
    Input('reset_selection', 'n_clicks'),
    Input('filter_bubble_comparison', 'n_clicks'),
    Input('reset_filter_duration_comparison', 'n_clicks'),
    Input('participant_anchor_comp', 'value'),
    State('comparison_plot', 'figure'),
    State('filter_bubble_duration_comparison', 'value'),
)
def comparison_plot_control(task_selection_comp, attribute_selection_comp, color_switch_comp, task_aligned, data,
                            reset_selection, filter_bubble_comparison_click, reset_filter_duration_comparison_click,
                            participant_anchor_comp,
                            current_figure, filter_bubble_duration_comparison,
                            ):
    global similarity_index_list
    global task_score_df
    global task_scores_df

    task_index = [int(word) for word in task_selection_comp.split() if word.isdigit()]
    task_index = task_index[0] - 1

    task_score = task_score_DTW[task_index][participant_anchor_comp - 1]
    task_scores_df = task_score_DTW[task_index]
    task_score_df = pd.DataFrame(task_score, columns=['score'])
    task_score_df.insert(0, 'participant', task_score_df.index + 1)

    similarity_index_list = sorted(range(len(task_score)), key=lambda x: task_score[x], reverse=True)
    # Re-position the selected participant to the first index, while keeping the order
    similarity_index_list.remove(participant_anchor_comp - 1)
    similarity_index_list.insert(0, participant_anchor_comp - 1)

    # Constants
    attr_default_color = ['rgba(0, 95, 96,1)']
    fig = make_subplots(shared_xaxes=True, rows=num_of_participant, cols=1, vertical_spacing=0.01,
                        specs=[[{"secondary_y": True}]] * num_of_participant,
                        y_title='Participants',  # x_title='Time',
                        )
    ################################################################################################
    # For if a bubble is focused/clicked on
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == 'comparison_plot':
        curve_number = data["points"][0]['curveNumber']
        traces = current_figure['data']

        opacity = [1 if t == curve_number else 0.2 for t in range(len(traces))]
        for t, op in enumerate(opacity):
            current_figure['data'][t]['marker']['opacity'] = op

        return current_figure, None

    ################################################################################################
    # Else, normally
    height_factor = 0
    label_encoder = LabelEncoder()
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
    default_color = ['rgba(0, 95, 96,1)']

    df_interaction_list_sorted = reorder_list_by_indexes(df_interaction_list, similarity_index_list)

    for p in range(0, num_of_participant):
        # Preprocess and draw bubble DF
        bubble_df = df_interaction_list_sorted[p]
        attr_df = df_interaction_list_sorted[p]
        attr_df_plot = attr_df.copy()

        # Slice the DF based on the selected task
        current_task = int(task_selection_comp[-1])
        next_task = 'Task ' + str(current_task + 1)

        task_start = attr_df_plot[attr_df_plot['id'] == str(task_selection_comp)]['start'].iloc[0]

        start_index_bubble = bubble_df[bubble_df.id == task_selection_comp].index[0]
        end_index_bubble = bubble_df[bubble_df.id == next_task].index[0]
        bubble_df = bubble_df.loc[start_index_bubble:end_index_bubble]

        start_index_attr = attr_df_plot[attr_df_plot.id == task_selection_comp].index[0]
        end_index_attr = attr_df_plot[attr_df_plot.id == next_task].index[0]
        attr_df_plot = attr_df_plot.loc[start_index_attr:end_index_attr]

        # TODO: testing the duration filter on the attribute entry
        if filter_bubble_duration_comparison is not None:
            attr_df_plot = attr_df_plot[attr_df_plot['duration_total'] >= filter_bubble_duration_comparison]
            attr_df_plot.reset_index(drop=True, inplace=True)

        # Align the comparison views
        if task_aligned:
            align_datetime = datetime.strptime('2023-05-04 00:00:00', '%Y-%m-%d %H:%M:%S')

            datetime_offset_bubble = bubble_df['start'].iloc[0] - align_datetime
            bubble_df_aligned = bubble_df.copy()
            bubble_df_aligned['start'] = bubble_df['start'] - datetime_offset_bubble
            bubble_df_aligned['end'] = bubble_df['end'] - datetime_offset_bubble
            bubble_df = bubble_df_aligned.copy()

            datetime_offset_attr = attr_df_plot['start'].iloc[0] - align_datetime
            attr_df_plot_aligned = attr_df_plot.copy()
            attr_df_plot_aligned['start'] = attr_df_plot['start'] - datetime_offset_attr
            attr_df_plot_aligned['end'] = attr_df_plot['end'] - datetime_offset_attr
            attr_df_plot = attr_df_plot_aligned.copy()

            task_start = attr_df_plot['start'].iloc[0]

        attr_df_plot = attr_df_plot[-attr_df_plot['id'].isin(tasks)]

        attr_df_plot['year_norm'] = attr_df_plot['year'].apply(normalize_year_attr_system)

        attr_df_plot['gender_encoded'] = label_encoder.fit_transform(attr_df_plot['gender'])
        attr_df_plot['gender_norm'] = attr_df_plot['gender_encoded'].apply(normalize_gender_attr)
        attr_df_plot['year_norm'] = attr_df_plot['year_norm'].astype(float)
        attr_df_plot['gender_norm'] = attr_df_plot['gender_norm'].astype(float)

        last_row = attr_df_plot.iloc[-1].copy()
        temp_last_row_dict = last_row.drop('end').to_dict()
        temp_last_row_dict['start'] = last_row['end']
        temp_last_row_df = pd.DataFrame([temp_last_row_dict])
        attr_df_plot = pd.concat([attr_df_plot, temp_last_row_df], ignore_index=True)

        ###############################################################
        # Add the bubbles as the base
        bubble_df = bubble_df[-bubble_df['id'].isin(tasks)]

        # TODO: testing the duration filter on the attribute entry
        if filter_bubble_duration_comparison is not None:
            bubble_df = bubble_df[bubble_df['duration_total'] >= filter_bubble_duration_comparison]
            bubble_df.reset_index(drop=True, inplace=True)

        custom_data = ['id', 'view', 'duration', 'municipality']
        hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                         'View: %{customdata[1]}<br>' \
                         'Start: %{base|%M:%S}<br>' \
                         'End: %{x|%M:%S}<br>' \
                         'Duration: %{customdata[2]} s<br>' \
                         'Municipality: %{customdata[3]}<br>' \
                         '<extra></extra>'

        # Options for bubble BG color
        # If colored by 'view'
        if color_switch_comp == 'view':
            bg_trace = (px.timeline(bubble_df, x_start='start', x_end='end', y='participant', opacity=0.5,
                                    color='view', color_discrete_map=color_mapping, custom_data=custom_data,
                                    pattern_shape='category', pattern_shape_map=pattern_shape_map)
                        .update_traces(hovertemplate=hover_template))
        # If colored by 'municipality'
        elif color_switch_comp == 'municipality':
            bg_trace = (px.timeline(bubble_df, x_start='start', x_end='end', y='participant', opacity=0.5,
                                    color='municipality', color_discrete_map=color_mapping_municipality,
                                    custom_data=custom_data,
                                    pattern_shape='category', pattern_shape_map=pattern_shape_map)
                        .update_traces(hovertemplate=hover_template))
        # If unified color
        else:
            bg_trace = (px.timeline(bubble_df, x_start='start', x_end='end', y='participant', opacity=0.5,
                                    color='view', color_discrete_sequence=default_color, custom_data=custom_data,
                                    pattern_shape='category', pattern_shape_map=pattern_shape_map)
                        .update_traces(hovertemplate=hover_template))

        for trace in bg_trace.data:
            fig.add_trace(trace, row=p + 1, col=1, secondary_y=False)

        ###############################################################
        # Add additional plots
        if 'show_gender_comp' in attribute_selection_comp:
            gender_custom_data = ['gender', 'end', 'duration']
            gender_hover_template = '<b>Gender: %{customdata[0]}</b><br><br>' \
                                    'Start: %{x|%M:%S}<br>' \
                                    'End: %{customdata[1]|%M:%S}<br>' \
                                    'Duration: %{customdata[2]} s<br>' \
                                    '<extra></extra>'
            gender_trace = (px.line(attr_df_plot, x='start', y='gender_norm', line_shape='hv',
                                    color_discrete_sequence=attr_default_color,
                                    custom_data=gender_custom_data)
                            .update_traces(hovertemplate=gender_hover_template, mode='lines', line_width=2).data)
            # use 'lines+markers' to add the dot
            for trace in gender_trace:
                fig.add_trace(trace, row=p + 1, col=1, secondary_y=True)

        if 'show_year_comp' in attribute_selection_comp:
            gender_custom_data = ['year', 'end', 'duration']
            gender_hover_template = '<b>Year: %{customdata[0]}</b><br><br>' \
                                    'Start: %{x|%M:%S}<br>' \
                                    'End: %{customdata[1]|%M:%S}<br>' \
                                    'Duration: %{customdata[2]} s<br>' \
                                    '<extra></extra>'
            year_trace = (px.line(attr_df_plot, x='start', y='year_norm', line_shape='hv',
                                  color_discrete_sequence=attr_default_color,
                                  custom_data=gender_custom_data)
                          .update_traces(hovertemplate=gender_hover_template, mode='lines', line_width=2).data)
            for trace in year_trace:
                fig.add_trace(trace, row=p + 1, col=1, secondary_y=True)
                if 'show_gender_comp' in attribute_selection_comp:
                    fig.add_hline(y=-0.5, row=p + 1, col=1, line_dash='dash', line_color='white', secondary_y=True)

        # Add vertical line for task starting point
        fig.add_vline(x=task_start, row=p + 1, col=1, line_width=2, line_dash="dot", line_color="black")

    # Update secondary y-axis
    fig.update_yaxes(range=[-2.3, 1.3], secondary_y=True, showticklabels=False)

    # Overall styling of the plot
    fig.update_xaxes(type='date', linecolor='black', linewidth=3, showticklabels=False)

    fig.update_layout(showlegend=False,
                      barmode='overlay',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin={'l': 65, 'r': 0, 't': 0, 'b': 5},
                      bargap=0,
                      height=int(comp_plot_height),
                      )

    return fig, None


###########################################################################################
# Individual view controls
###########################################################################################

# Style switch for the 'plot type' options
@app.callback(
    Output('bubble_color_switch', 'options'),
    Output('bubble_color_switch', 'style'),
    Output('bubble_color_switch', 'value'),
    Output('indi_attribute_selection', 'options'),
    Output('indi_attribute_selection', 'style'),
    Output('indi_attribute_selection', 'value'),
    Input('bubble_length_switch', 'value'),
)
def bubble_length_toggle(bubble_length_switch):
    (bubble_length_toggle_new_options, bubble_length_toggle_new_style, bubble_length_toggle_new_value,
     indi_attribute_selection_option, indi_attribute_selection_style, indi_attribute_selection_value) \
        = bubble_length_toggle_style(bubble_length_switch)
    return (bubble_length_toggle_new_options, bubble_length_toggle_new_style, bubble_length_toggle_new_value,
            indi_attribute_selection_option, indi_attribute_selection_style, indi_attribute_selection_value)


# Style switch for the duration filter option
@app.callback(
    Output('filter_bubble_duration_indi', 'value'),
    Input('reset_filter_duration_indi', 'n_clicks'),
)
def indi_duration_filter_toggle(reset_filter_duration_indi_click):
    button_id = ctx.triggered_id
    if button_id == 'reset_filter_duration_indi':
        return None


@app.callback(
    Output('timeline_plot', 'figure'),
    Input('participant_selection', 'value'),
    Input('task_selection_individual', 'value'),
    Input('bubble_length_switch', 'value'),
    Input('bubble_color_switch', 'value'),
    Input('indi_attribute_selection', 'value'),
    Input('filter_bubble_indi', 'n_clicks'),
    Input('reset_filter_duration_indi', 'n_clicks'),
    State('filter_bubble_duration_indi', 'value')
)
def timeline_plot_control(participant_selection, task_selection_individual, bubble_length_switch, bubble_color_switch,
                          indi_attribute_selection,
                          filter_bubble_individual_click, reset_filter_duration_indi_click,
                          filter_bubble_duration_indi):
    # Constants
    df_index = participant_selection - 1
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']

    height_factor = 0

    y_category_array = []

    # If reset the duration filter
    button_id = ctx.triggered_id
    if button_id == 'reset_filter_duration_indi':
        filter_bubble_duration_indi = None

    ################################################################################################
    ################################################################################################
    ################################################################################################
    # If using holistic view (complete bubbles)
    if bubble_length_switch == 'life':
        # Read the DF
        selected_df = df_bubble_list[df_index]
        # Read the tasks
        task_rows = selected_df[selected_df.id.str.contains('Task ')]

        custom_data = ['duration', 'view']
        hover_template = '<b>ID: %{y}</b><br>' \
                         'Start: %{base|%H:%M:%S}<br>' \
                         'End: %{x|%H:%M:%S}<br>' \
                         'Duration: %{customdata[0]} s<br>' \
                         'View: %{customdata[1]}<br>' \
                         '<extra></extra>'

        ################################################
        # If viewing all tasks
        if task_selection_individual == 'All':
            # Computer the x-axis range
            current_task_time = selected_df.loc[selected_df['id'] == 'Task 1', 'start'].values[0]
            next_task_time = selected_df.loc[selected_df['id'] == 'Task 6', 'start'].values[0]
            x_axis = [pd.to_datetime(current_task_time) - timedelta(seconds=0.15),
                      pd.to_datetime(next_task_time) + timedelta(seconds=0.15)]

            # Remove the 'task' entries
            selected_df_plot = selected_df[-selected_df['id'].isin(tasks)]
            # Filter the duration if desired
            if filter_bubble_duration_indi is not None:
                selected_df_plot = selected_df_plot[selected_df_plot['duration_total'] >= filter_bubble_duration_indi]
                selected_df_plot.reset_index(drop=True, inplace=True)

        ################################################
        # If viewing a specific task
        else:
            # Computer the x-axis range
            current_task_time = selected_df.loc[selected_df['id'] == task_selection_individual, 'start'].values[0]
            current_task = int(task_selection_individual[-1])
            next_task = 'Task ' + str(current_task + 1)
            next_task_time = selected_df.loc[selected_df['id'] == next_task, 'start'].values[0]
            x_axis = [pd.to_datetime(current_task_time) - timedelta(seconds=0.15),
                      pd.to_datetime(next_task_time) + timedelta(seconds=0.15)]

            # Same as removing the 'task' entries
            current_task_index = selected_df[selected_df.id == task_selection_individual].index[0]
            next_task_index = selected_df[selected_df.id == next_task].index[0]
            filtered_selected_interaction_df = selected_df.loc[current_task_index + 1:next_task_index - 1]

            selected_df_plot = filtered_selected_interaction_df

            # Filter the duration if desired
            if filter_bubble_duration_indi is not None:
                selected_df_plot = selected_df_plot[selected_df_plot['duration_total'] >= filter_bubble_duration_indi]
                selected_df_plot.reset_index(drop=True, inplace=True)

        bubble_id_list = selected_df_plot['id'].unique()
        bubble_id_list = sorted(bubble_id_list)
        y_category_array += bubble_id_list

        height_factor = len(selected_df_plot['id'].unique())

        # Plot
        default_color = ['rgba(0, 95, 96,1)']
        fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id',
                          color_discrete_sequence=default_color, custom_data=custom_data
                          ).update_traces(hovertemplate=hover_template).update_yaxes(title='ID')

    ################################################################################################
    ################################################################################################
    ################################################################################################
    # If using detailed view (interaction bubbles)
    else:
        # Read the DF
        selected_df = df_interaction_list[df_index]
        # Read the tasks
        task_rows = selected_df[selected_df.id.str.contains('Task ')]

        custom_data = ['duration', 'year', 'gender', 'municipality', 'duration_total']
        hover_template = '<b>ID: %{y}</b><br>' \
                         'Start: %{base|%M:%S}<br>' \
                         'End: %{x|%M:%S}<br>' \
                         'Duration: %{customdata[0]} s<br>' \
                         'Duration total: %{customdata[4]} s<br><br>' \
                         '<b>Attributes</b><br>' \
                         'Year: %{customdata[1]}<br>' \
                         'Gender: %{customdata[2]}<br>' \
                         'Municipality: %{customdata[3]}<br>' \
                         '<extra></extra>'

        ################################################
        # If viewing all tasks
        if task_selection_individual == 'All':
            # Bubble color options
            # If viewing unified color
            if bubble_color_switch == 'unified':  # color by 'participant'
                color = 'participant'
                default_color = ['rgba(0, 95, 96,1)']
            else:
                color = 'municipality'
                default_color = color_mapping_municipality  # px.colors.sequential.tempo_r
            attr_default_color = ['rgba(0, 95, 96,1)']

            current_task_time = selected_df.loc[selected_df['id'] == 'Task 1', 'start'].values[0]
            next_task_time = selected_df.loc[selected_df['id'] == 'Task 6', 'start'].values[0]
            x_axis = [pd.to_datetime(current_task_time) - timedelta(seconds=0.15),
                      pd.to_datetime(next_task_time) + timedelta(seconds=0.15)]

            # Remove the 'task' entries
            selected_df_plot = selected_df[-selected_df['id'].isin(tasks)]

            # Filter the duration if desired
            if filter_bubble_duration_indi is not None:
                selected_df_plot = selected_df_plot[selected_df_plot['duration_total'] > filter_bubble_duration_indi]
                selected_df_plot.reset_index(drop=True, inplace=True)

            bubble_id_list = selected_df_plot['id'].unique()
            bubble_id_list = sorted(bubble_id_list)
            y_category_array += bubble_id_list
            height_factor = len(selected_df_plot['id'].unique())

            # If no additional plot - only showing bubbles
            if not indi_attribute_selection:
                if bubble_color_switch == 'unified':
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_sequence=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template).update_yaxes(title='ID')

                else:
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_map=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template).update_yaxes(title='ID')

            # If an attribute is added
            else:
                # Add an artificial entry before each task to make the plot complete
                task_2_6 = ['Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
                attr_df_plot = pd.DataFrame(columns=selected_df.columns)

                for index, row in selected_df.iterrows():
                    attr_df_plot = pd.concat([attr_df_plot, row.to_frame().T], ignore_index=True)

                    if int(index) < len(selected_df) - 1:
                        next_row = selected_df.loc[index + 1]
                        if next_row['id'] in task_2_6:
                            last_row_temp = row.copy()
                            last_row_temp['start'] = row['end']
                            last_row_temp['end'] = ''
                            last_row_temp['duration'] = ''
                            attr_df_plot = pd.concat([attr_df_plot, last_row_temp.to_frame().T],
                                                     ignore_index=True)

                if filter_bubble_duration_indi is not None:
                    # Remove the empty entry
                    attr_df_plot = attr_df_plot[attr_df_plot['duration'] != '']
                    # Convert to int type
                    attr_df_plot['duration'] = attr_df_plot['duration'].astype(int)

                    attr_df_plot = attr_df_plot[attr_df_plot['duration_total'] > filter_bubble_duration_indi]
                    attr_df_plot.reset_index(drop=True, inplace=True)

                attr_trace = []
                y_title = 'ID'

                # Add the bubbles as the base
                if bubble_color_switch == 'unified':
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_sequence=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template)

                else:
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_map=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template)

                # Add additional plots
                attr_df_plot = attr_df_plot[-attr_df_plot['id'].isin(tasks)]

                if 'show_gender_individual' in indi_attribute_selection:
                    gender_custom_data = ['id', 'gender']
                    gender_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                            'Gender: %{customdata[1]}<br>' \
                                            '<extra></extra>'

                    gender_trace = (px.line(attr_df_plot, x='start', y='gender', line_shape='hv',
                                            color_discrete_sequence=attr_default_color,
                                            custom_data=gender_custom_data)
                                    .update_traces(hovertemplate=gender_hover_template, mode='lines+markers').data)
                    attr_trace += gender_trace
                    y_title += ' | Gender'

                    # Calculate the height factor
                    gender_list = attr_df_plot['gender'].unique()
                    gender_list = [x for x in gender_list if str(x) != 'nan']
                    gender_list = sorted(gender_list)
                    y_category_array += gender_list
                    height_factor += len(gender_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(gender_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                if 'show_year_individual' in indi_attribute_selection:
                    year_custom_data = ['id', 'year']
                    year_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                          'Year: %{customdata[1]}<br>' \
                                          '<extra></extra>'
                    year_trace = (px.line(attr_df_plot, x='start', y='year', line_shape='hv',
                                          color_discrete_sequence=attr_default_color,
                                          custom_data=year_custom_data)
                                  .update_traces(hovertemplate=year_hover_template, mode='lines+markers').data)
                    attr_trace += year_trace
                    y_title += ' | Year'

                    # Calculate the height factor
                    year_list = attr_df_plot['year'].unique()
                    year_list = [x for x in year_list if str(x) != '']
                    min_year = int(min(year_list))
                    max_year = int(max(year_list))
                    year_list = [str(year) for year in range(min_year, max_year + 1)]
                    y_category_array += year_list
                    height_factor += len(year_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(year_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                if 'show_municipality_individual' in indi_attribute_selection:
                    municipality_custom_data = ['id', 'municipality']
                    municipality_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                                  'Municipality: %{customdata[1]}<br>' \
                                                  '<extra></extra>'
                    municipality_trace = (px.line(attr_df_plot, x='start', y='municipality', line_shape='hv',
                                                  color_discrete_sequence=attr_default_color,
                                                  custom_data=municipality_custom_data)
                                          .update_traces(hovertemplate=municipality_hover_template,
                                                         mode='lines+markers').data)
                    attr_trace += municipality_trace
                    y_title += ' | Municipality'

                    # Calculate the height factor
                    municipality_list = attr_df_plot['municipality'].unique()
                    municipality_list = [x for x in municipality_list if str(x) != 'nan']
                    municipality_list = sorted(municipality_list)
                    y_category_array += municipality_list
                    height_factor += len(municipality_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(municipality_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                for trace in attr_trace:
                    fig.add_trace(trace)

                fig.update_yaxes(title=y_title)

        ################################################
        # If viewing a specific task
        else:
            # Bubble color options
            # If viewing unified color
            if bubble_color_switch == 'unified':  # color by 'participant'
                color = 'participant'
                default_color = ['rgba(0, 95, 96,1)']
            # If colored by municipality
            else:
                color = 'municipality'
                default_color = color_mapping_municipality  # px.colors.sequential.tempo_r
            attr_default_color = ['rgba(0, 95, 96,1)']

            # Computer the x-axis range
            current_task_time = selected_df.loc[selected_df['id'] == task_selection_individual, 'start'].values[0]
            current_task = int(task_selection_individual[-1])
            next_task = 'Task ' + str(current_task + 1)
            next_task_time = selected_df.loc[selected_df['id'] == next_task, 'start'].values[0]
            x_axis = [pd.to_datetime(current_task_time) - timedelta(seconds=0.15),
                      pd.to_datetime(next_task_time) + timedelta(seconds=0.15)]

            # Same as removing the 'task' entries
            current_task_index = selected_df[selected_df.id == task_selection_individual].index[0]
            next_task_index = selected_df[selected_df.id == next_task].index[0]
            selected_df_plot = selected_df.loc[current_task_index + 1:next_task_index - 1]

            if filter_bubble_duration_indi is not None:
                selected_df_plot = selected_df_plot[selected_df_plot['duration_total'] > filter_bubble_duration_indi]
                selected_df_plot.reset_index(drop=True, inplace=True)

            bubble_id_list = selected_df_plot['id'].unique()
            bubble_id_list = sorted(bubble_id_list)
            y_category_array += bubble_id_list

            # If no additional plot - only showing bubbles in this task
            if not indi_attribute_selection:
                if bubble_color_switch == 'municipality':  # color by 'participant'
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_map=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template).update_yaxes(title='ID')
                else:
                    fig = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                      color_discrete_sequence=default_color, custom_data=custom_data,
                                      pattern_shape='category', pattern_shape_map=pattern_shape_map
                                      ).update_traces(width=1, hovertemplate=hover_template).update_yaxes(title='ID')
                height_factor += len(selected_df_plot['id'].unique())

            # If an attribute is added in this task
            else:
                # Add an artificial entry before each task to make the plot complete
                task_2_6 = ['Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
                attr_df_plot = pd.DataFrame(columns=selected_df.columns)

                for index, row in selected_df.iterrows():
                    attr_df_plot = pd.concat([attr_df_plot, row.to_frame().T], ignore_index=True)

                    if int(index) < len(selected_df) - 1:
                        next_row = selected_df.loc[index + 1]
                        if next_row['id'] in task_2_6:
                            last_row_temp = row.copy()
                            last_row_temp['start'] = row['end']
                            last_row_temp['end'] = ''
                            last_row_temp['duration'] = ''
                            # print(last_row_temp)
                            attr_df_plot = pd.concat([attr_df_plot, last_row_temp.to_frame().T],
                                                     ignore_index=True)

                if filter_bubble_duration_indi is not None:
                    # Remove the empty entry
                    attr_df_plot = attr_df_plot[attr_df_plot['duration'] != '']
                    # Convert to int type
                    attr_df_plot['duration'] = attr_df_plot['duration'].astype(int)

                    attr_df_plot = attr_df_plot[attr_df_plot['duration_total'] >= filter_bubble_duration_indi]
                    attr_df_plot.reset_index(drop=True, inplace=True)

                # TODO: add back the last row to make the line extend to the end (to make it complete)
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
                fig.update_xaxes(type='date')
                attr_trace = []
                y_title = 'ID'

                # Add the bubbles as the base
                if bubble_color_switch == 'municipality':  # color by 'participant'
                    bubble_trace = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                               color_discrete_map=default_color, custom_data=custom_data,
                                               pattern_shape='category', pattern_shape_map=pattern_shape_map
                                               ).update_traces(width=1, hovertemplate=hover_template)
                else:
                    bubble_trace = px.timeline(selected_df_plot, x_start='start', x_end='end', y='id', color=color,
                                               color_discrete_sequence=default_color, custom_data=custom_data,
                                               pattern_shape='category', pattern_shape_map=pattern_shape_map
                                               ).update_traces(width=1, hovertemplate=hover_template)

                for trace in bubble_trace.data:
                    fig.add_trace(trace, 1, 1)

                height_factor = len(selected_df_plot['id'].unique())

                # Add additional plots
                if 'show_gender_individual' in indi_attribute_selection:
                    gender_custom_data = ['id', 'gender']
                    gender_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                            'Gender: %{customdata[1]}<br>' \
                                            '<extra></extra>'
                    current_task_index = attr_df_plot[attr_df_plot.id == task_selection_individual].index[0]
                    next_task_index = attr_df_plot[attr_df_plot.id == next_task].index[0]
                    gender_df_plot = attr_df_plot.loc[current_task_index + 1:next_task_index - 1]
                    gender_trace = (px.line(gender_df_plot, x='start', y='gender', line_shape='hv',
                                            color_discrete_sequence=attr_default_color,
                                            custom_data=gender_custom_data)
                                    .update_traces(hovertemplate=gender_hover_template).data)
                    attr_trace += gender_trace
                    y_title += ' | Gender'

                    # Calculate the height factor
                    gender_list = gender_df_plot['gender'].unique()
                    gender_list = [x for x in gender_list if str(x) != 'nan']
                    gender_list = sorted(gender_list)
                    y_category_array += gender_list
                    height_factor += len(gender_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(gender_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                if 'show_year_individual' in indi_attribute_selection:
                    year_custom_data = ['id', 'year']
                    year_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                          'Year: %{customdata[1]}<br>' \
                                          '<extra></extra>'
                    current_task_index = attr_df_plot[attr_df_plot.id == task_selection_individual].index[0]
                    next_task_index = attr_df_plot[attr_df_plot.id == next_task].index[0]
                    year_df_plot = attr_df_plot.loc[current_task_index + 1:next_task_index - 1]

                    year_trace = (px.line(year_df_plot, x='start', y='year', line_shape='hv',
                                          color_discrete_sequence=attr_default_color,
                                          custom_data=year_custom_data)
                                  .update_traces(hovertemplate=year_hover_template).data)
                    attr_trace += year_trace
                    y_title += ' | Year'

                    # Calculate the height factor
                    year_list = year_df_plot['year'].unique()
                    year_list = [x for x in year_list if str(x) != 'nan']
                    min_year = int(min(year_list))
                    max_year = int(max(year_list))
                    year_list = [str(year) for year in range(min_year, max_year + 1)]
                    y_category_array += year_list
                    height_factor += len(year_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(year_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                if 'show_municipality_individual' in indi_attribute_selection:
                    municipality_custom_data = ['id', 'municipality']
                    municipality_hover_template = '<b>ID: %{customdata[0]}</b><br>' \
                                                  'Municipality: %{customdata[1]}<br>' \
                                                  '<extra></extra>'
                    current_task_index = attr_df_plot[attr_df_plot.id == task_selection_individual].index[0]
                    next_task_index = attr_df_plot[attr_df_plot.id == next_task].index[0]
                    municipality_df_plot = attr_df_plot.loc[current_task_index + 1:next_task_index - 1]

                    municipality_trace = (px.line(municipality_df_plot, x='start', y='municipality', line_shape='hv',
                                                  color_discrete_sequence=attr_default_color,
                                                  custom_data=municipality_custom_data)
                                          .update_traces(hovertemplate=municipality_hover_template).data)
                    attr_trace += municipality_trace
                    y_title += ' | Municipality'

                    # Calculate the height factor
                    municipality_list = municipality_df_plot['municipality'].unique()
                    municipality_list = [x for x in municipality_list if str(x) != 'nan']
                    municipality_list = sorted(municipality_list)
                    y_category_array += municipality_list
                    height_factor += len(municipality_list)

                    # Add a horizontal line to separate
                    sep_line = height_factor - len(municipality_list) - 0.5
                    fig.add_hline(y=sep_line, line_dash='dot', line_color='darkGray')

                for trace in attr_trace:
                    fig.add_trace(trace, 1, 1)
                fig.update_yaxes(title=y_title)

    ################################################
    # Plot controls end; figure config start
    ################################################
    # Adjust the y-axis
    fig.update_yaxes(type='category',
                     categoryarray=y_category_array)

    # Add view change background
    bg_df = df_vc_list[df_index]
    bg_df = bg_df[-bg_df['id'].isin(tasks)]
    for row in bg_df.itertuples():
        view = row.view
        fillcolor = color_mapping.get(view, 'rgba(0,0,0,0)')  # Default to transparent

        fig.add_vrect(
            x0=row.start, x1=row.end,
            fillcolor=fillcolor, opacity=0.12,
            layer="below", line_width=0,
        )

    # Add vertical lines
    # Extract the 'task' rows
    tasks_timestamp = task_rows.start.tolist()
    for task in tasks_timestamp:
        fig.add_vline(
            x=task,
            line_width=2, line_dash="dash", line_color="black",
        )
        fig.add_annotation(
            x=task,
            y=-0.5,  # Adjust the Y-coordinate as needed
            text=f"<b>{task.minute:02d}:{task.second:02d}</b>",
            ax=25,
            ay=15,
            font=dict(color='white'),
            bgcolor="cadetBlue",
        )

    # Customize appearance
    ind_min_height = 50
    ind_height_scale = 15
    ind_height = ind_min_height + height_factor * ind_height_scale

    fig.update_layout(showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin={'l': 0, 'r': 50, 't': 0, 'b': 5},
                      bargap=0,
                      height=int(ind_height)
                      )
    fig.update_xaxes(showticklabels=False, spikemode='toaxis', range=x_axis, showgrid=False)
    fig.update_yaxes(spikemode='toaxis')
    return fig


# --------------------------------------------------------------------------
# start the app
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
