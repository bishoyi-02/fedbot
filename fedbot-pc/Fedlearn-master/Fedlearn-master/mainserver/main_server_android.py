from flask import Flask, request,send_file
from flask_socketio import SocketIO
import requests, json
import ast
import base64
# from fl_agg import model_aggregation

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def hello():
	return "Server running !"

@app.route('/clientstatus', methods=['GET','POST'])
def client_status():
	# url = "http://localhost:8001/serverack"
	if request.method == 'POST':
		# print(request.remote_addr)
		# client_id = request.json['client_id']
		client_id=request.remote_addr
		with open('clients.txt', 'a+') as f:
			f.write("http://"+client_id+ '/\n')

		# print(client_id)

		if client_id:
			serverack = "Server Acknowledged"
			# response = requests.post( url, data=json.dumps(serverack), headers={'Content-Type': 'application/json'} )
			return str(serverack)
		else:
			return "Client status not OK!"
	else:
		return "Client GET request received!"
		
@app.route('/secagg_model', methods=['POST'])
def get_secagg_model():
	if request.method == 'POST':
		file = request.files['model'].read()
		fname = request.files['json'].read()
		# cli = request.files['id'].read()

		fname = ast.literal_eval(fname.decode("utf-8"))
		cli = fname['id']+'\n'
		fname = fname['fname']

		# with open('clients.txt', 'a+') as f:
		# 	f.write(cli)
		
		# print(fname, cli)
		wfile = open("agg_model/"+fname, 'wb')
		wfile.write(file)
			
		return "Model received!"
	else:
		return "No file received!"

# @app.route('/aggregate_models')
# def perform_model_aggregation():
# 	model_aggregation()
# 	return 'Model aggregation done!\nGlobal model written to persistent storage.'

# @app.route('/send_model_clients')
# def send_agg_to_clients():
# 	clients = ''
# 	with open('clients.txt', 'r') as f:
# 		clients = f.read()
# 	clients = clients.split('\n')
	
# 	for c in clients:
# 		if c != '':
# 			file = open("agg_model/agg_model.h5", 'rb')
# 			data = {'fname':'agg_model.h5'}
# 			files = {
# 				'json': ('json_data', json.dumps(data), 'application/json'),
# 				'model': ('agg_model.h5', file, 'application/octet-stream')
# 			}
			
# 			print(c+'aggmodel')
# 			# req = requests.post(url=c+'aggmodel', files=files)
# 			req = requests.post(url=c, files=files)
# 			print(req.status_code)
	
# 	# print(req.text)
# 	return "Aggregated model sent !"

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "./agg_model/agg_model.h5"
    return send_file(path, as_attachment=True)



if __name__ == '__main__':
	socketio.run(app,host='0.0.0.0', port=8000, debug=False, use_reloader=False)















