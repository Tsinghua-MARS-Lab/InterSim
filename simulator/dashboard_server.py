from flask import Flask
from flask import request, jsonify
import json, os, util, pickle
app = Flask(__name__)
SIMULATION_RESULT_PATH = './sim_result'

from flask_cors import CORS
CORS(app)

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

@app.route("/")
def hello_world():
    return "<p>InterSim Beta Server is running.</p>"


@app.route('/vis', methods=['GET', 'POST'])
def visualization_get():
    sim_name = request.args.get('sim')
    scene_id = request.args.get('sceneid')
    file_id = request.args.get('fileid')
    if request.method == 'GET':
        with open(f"sim_result/{sim_name}/json/{file_id}.json", "r") as json_file:
            my_dict = json.load(json_file)
            keys = list(my_dict.keys())
            my_dict['selected_scene_index'] = keys.index(scene_id)
        return jsonify(my_dict)

@app.route('/list_scenarios')
def get_simulations():
    return summary_simulations()

@app.route('/list_simulation_selection')
def get_simulation_selection():
    dataset = request.args.get('dataset')
    return list_simulation_selection(dataset=dataset)

@app.route('/list_senarios')
def get_scenarios_list():
    simulation_name = request.args.get('simulation')
    return list_scenarios(simulation_name)

def check_path_valid(path):
    if not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, 'sim.info')):
        return False
    return True

def summary_simulations(path=SIMULATION_RESULT_PATH):
    html_str = ""
    # loop all simulations and load their info and add to the html table
    for each_path in os.listdir(path):
        if not check_path_valid(os.path.join(path, each_path)):
            continue
        sim_info = load(os.path.join(path, each_path, 'sim.info'))
        # dataset_with_map = '-' + sim_info['map_info'] if 'map_info' in sim_info and sim_info['map_info'] is not None else ''
        # dataset_with_map = sim_info['dataset'] + dataset_with_map
        dataset_with_map = sim_info['dataset']
        html_str += f"<tr><td>{sim_info['name']}</td>" \
                    f"<td>{sim_info['task']}</td>" \
                    f"<td>{dataset_with_map}</td>" \
                    f"<td>{sim_info['planner']}</td>" \
                    f"<td>{sim_info['predictor']}</td>" \
                    f"<td>{sim_info['status']}</td>" \
                    f"<td>{sim_info['starting_time']}</td>"
        html_str += f"<td>{sim_info['ending_time']}</td>" if sim_info['ending_time'] is not None else "<td>-</td>"
        # add action drop
        html_str += f'''
        <td>
            <div class="dropdown">
              <a class="dropdown-toggle icon-burger-mini" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown"
                aria-haspopup="true" aria-expanded="false">
              </a>
              <div class="dropdown-menu dropdown-menu-right" aria-labelledby="dropdownMenuLink">
                <a class="dropdown-item" href="list.html?task={sim_info['task']}&dataset={dataset_with_map}&sim={each_path}">Detail</a>
              </div>
            </div>
          </td>
          '''
        html_str += "</tr>"
    return html_str

def list_simulation_selection(dataset=None, path=SIMULATION_RESULT_PATH):
    if dataset is None:
        return
    html_str = ""
    for each_path in os.listdir(path):
        if not check_path_valid(os.path.join(path, each_path)):
            continue
        if dataset not in each_path:
            continue
        html_str += f"<option value=\"{each_path}\">{each_path}</option>"
    return html_str

def list_scenarios(simulation_name=None):
    if simulation_name is None:
        return
    simulation_path = os.path.join(SIMULATION_RESULT_PATH, simulation_name)
    if not os.path.exists(simulation_path):
        return
    html_str = """
    <thead>
        <tr>
          <th>Scenario id</th>
          <th>Collsion Rate</th>
          <th>Progress (m)</th>
          <th></th>
        </tr>
      </thead>
      <tbody id="scenario_list">
      """
    for each_playback_path in os.listdir(os.path.join(simulation_path,  'playback')):
        loaded_playback = load(os.path.join(simulation_path, 'playback', each_playback_path))
        for each_scenario_id in loaded_playback:
            html_str += "<tr class=\"list-group-item-action\">"
            metric_rst = loaded_playback[each_scenario_id]['metrics']
            task = loaded_playback[each_scenario_id]['info']['task']
            dataset = loaded_playback[each_scenario_id]['info']['dataset']
            collision_rate = len(metric_rst['collided_pairs'])
            progress = metric_rst['progress']
            # jerk = "N/A"  # metric_rst['jerk']
            html_str += f"""
            <td>{each_scenario_id}</td>
            <td>{collision_rate}</td>
            <td>{progress}</td>
            """
            file_name = each_playback_path.split('.playback')[0]
            html_str += f"""
                <td>
                    <div class="dropdown">
                      <a class="dropdown-toggle icon-burger-mini" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown"
                        aria-haspopup="true" aria-expanded="false">
                      </a>
                      <div class="dropdown-menu dropdown-menu-right" aria-labelledby="dropdownMenuLink">
                        <a class="dropdown-item" href="visualization-detail.html?task={task}&sim={simulation_name}&fileid={file_name}&sceneid={each_scenario_id}">Visualize</a>
                      </div>
                    </div>
                  </td>
                </tr>
                """
    html_str += """</tbody>
                    </table>"""

    return html_str










