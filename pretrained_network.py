import numpy as np
import tensorflow as tf
import cv2
import json

from coco_class_info import coco_labels
from arg_parser import parser

tf.logging.set_verbosity(tf.logging.INFO)

def load_model_from_ckpt(model_path):
    print("Using the model {}".format(model_path))
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

    return sess, graph

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.config_json:
        
        with open(args.config_json, 'r') as f:
            config = json.load(f)
        
        print("Got the args {}".format(config))
        sess, graph = load_model_from_ckpt(config["model_path"])
        cap = cv2.VideoCapture(1)

        while(True):
            # Capture frame-by-frame
            ret, image = cap.read()
            pred = sess.run(config["output_nodes"], feed_dict={ config["input_node"]: np.reshape(image, (1, image.shape[0], image.shape[1], 3)) } )
            
            #every returned element is an array
            num_detections = int(pred[0][0])
            detected_classes = pred[1][0]
            detected_scores = pred[2][0]
            detected_boxes = pred[3][0]

            for idx, box in enumerate(detected_boxes[0:num_detections]):
                
                image_width = image.shape[1]
                image_height = image.shape[0]
                y1, x1, y2, x2 = box # the values are in the range 0 to 1.

                class_idx = detected_classes[idx]
                predicted_class = next((class_info for class_info in coco_labels if class_info["id"] == class_idx ), {})
                print(predicted_class)
                # converting the values from 0 to 1 to pixel values.
                x1 = int(x1*image_width)
                y1 = int(y1*image_height)
                x2 = int(x2*image_width)
                y2 = int(y2*image_height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(image,predicted_class["name"],(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)

            cv2.imshow("pred", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the graph info to log to learn about the architecture using tensorboard.
        writer = tf.summary.FileWriter('logs/graph', graph=graph)
        writer.close()

    else:
        print("Please pass the config.json file as args")