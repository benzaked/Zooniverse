from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from webargs.flaskparser import use_kwargs
from webargs import fields
from datetime import datetime
from Recommender import *
import logging

app = Flask(__name__)
api = Api(app)


@app.route('/algo', methods=['GET'])
def get():
    user_profile_id = request.params['user_profile_id']
    k = request.params['k']
    ip_address = request.params['ip_address']
    json_info = {'user_profile_id': user_profile_id,
                 'timestamp': str(datetime.datetime.now()),
                 'ip_address': ip_address}
    f = open("log_file_debugging.txt", "a")
    f.write(str(json_info) + ",")
    f.close()
    algorithm = map_user_algorithm(user_profile_id)
    print (user_profile_id, k, ip_address, algorithm)
    results = get_recommendations(user_profile_id, k, algorithm, ip_address)
    logging.info(str(
        {'user_profile_id': user_profile_id, 'timestamp': str(datetime.datetime.now()), 'algorithm': algorithm,
         'ip_address': ip_address, 'recommendations': results}))
    print(user_profile_id, ": ", results)
    return jsonify(dict(
        user_profile_id=user_profile_id,
        k=k,
        recommendations=[int(i) for i in results]
    ))


if __name__ == '__main__':
    import sys

    argv = sys.argv + [None, None]
    host = str(argv[1]) if argv[1] else '127.0.0.1'
    port = int(argv[2]) if argv[2] else 8080
    logging.basicConfig(filename='logger.log', level=logging.INFO)
    app.run(port=port, host=host)
