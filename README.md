# fashion mnist 수행하기

## fashion mnist 학습 모델 작성

- jupyter notebook terminal mode에서 fairing을 설치한다.

```
pip install kubeflow-fairing msrestazure
```

- 다음 학습 모델을 작성하여 fairing 할 수 있도록 한다.


```
import tensorflow as tf
import os
import argparse
from tensorflow.python.keras.callbacks import Callback


class MyFashionMnist(object):
  def train(self):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, type=float, default=0.001)
    parser.add_argument('--dropout_rate', required=False, type=float, default=0.3)
    parser.add_argument('--epoch', required=False, type=int, default=5)    
    parser.add_argument('--act', required=False, type=str, default='relu')        
    parser.add_argument('--layer', required=False, type=int, default=1)      
    parser.add_argument('--model_version', required=False, type=str, default='0001')    
    parser.add_argument('--checkpoint_dir', required=False, default='/reuslt/training_checkpoints')
    parser.add_argument('--saved_model_dir', required=False, default='/result/saved_model')        
    parser.add_argument('--tensorboard_log', required=False, default='/result/log')     
    args = parser.parse_args()    
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    for i in range(int(args.layer)):    
        model.add(tf.keras.layers.Dense(128, activation=args.act))
        if(i > 2) :
            model.add(tf.keras.layers.Dropout(args.dropout_rate))
        
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    
    model.summary()    
    
    # 체크포인트를 저장할 체크포인트 디렉터리를 지정합니다.
    checkpoint_dir = args.checkpoint_dir
    # 체크포인트 파일의 이름
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")        

    model.fit(x_train, y_train,
              verbose=0,
              validation_data=(x_test, y_test),
              epochs=args.epoch,
              callbacks=[KatibMetricLog(),
                        tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_log),
                        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                               save_weights_only=True)
                        ])
    
    path = args.saved_model_dir + "/" + args.model_version        
    model.save(path, save_format='tf')

    
class KatibMetricLog(Callback):
    def on_batch_end(self, batch, logs={}):
        print("batch=" + str(batch),
              "accuracy=" + str(logs.get('acc')),
              "loss=" + str(logs.get('loss')))
    def on_epoch_begin(self, epoch, logs={}):
        print("epoch " + str(epoch) + ":")
    
    def on_epoch_end(self, epoch, logs={}):
        print("Validation-accuracy=" + str(logs.get('val_acc')),
              "Validation-loss=" + str(logs.get('val_loss')))
        return
    
    
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow import fairing
        from kubeflow.fairing.kubernetes import utils as k8s_utils

        DOCKER_REGISTRY = 'ykkim77'
        fairing.config.set_builder(
            'append',
            image_name='fairing-job',
            base_image='brightfly/kubeflow-jupyter-lab:tf2.0-cpu',
            registry=DOCKER_REGISTRY, 
            push=True)
        # cpu 2, memory 5GiB
        fairing.config.set_deployer('job',
                                    namespace='practice',
                                    pod_spec_mutators=[
                                        k8s_utils.mounting_pvc(pvc_name="efs-claim", 
                                                              pvc_mount_path="/result"),
                                        k8s_utils.get_resource_mutator(cpu=2,
                                                                       memory=5)]
         
                                   )
        fairing.config.run()
    else:
        remote_train = MyFashionMnist()
        remote_train.train()
```


## katib로 하이퍼 파라미터 조정하기
- 다음 명세 파일을 이용하여 fairing 된 컨테이너를 기반으로 하이퍼 파라미터를 조정한다.

```
apiVersion: "kubeflow.org/v1beta1"
kind: Experiment
metadata:
  namespace: practice
  name: random-example
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: Validation-accuracy
    additionalMetricNames:
      - Train-accuracy
  algorithm:
    algorithmName: random
  parallelTrialCount: 3
  maxTrialCount: 12
  maxFailedTrialCount: 3
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace:
        min: "0.01"
        max: "0.03"
    - name: dropout_rate
      parameterType: double
      feasibleSpace:
        min: "0.1"
        max: "0.9"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: learning_rate
        description: Learning rate for the training model
        reference: learning_rate
      - name: dropout_rate
        description: dropout_rate
        reference: dropout_rate
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          metadata:
            annotations:
              sidecar.istio.io/inject: "false"
          spec:
            containers:
              - name: training-container
                image: ykkim77/fairing-job:38041ACE
                command:
                  - "python"
                  - "/app/fmniest_e2e.py"
                  - "--learning_rate=${trialParameters.learning_rate}"
                  - "--dropout_rate=${trialParameters.dropout_rate}"
            restartPolicy: Never
```


## service 코드 작성하기

- 모델 서빙을 위한 코드를 작성하고, 역시 fairing 하여 컨테이너화 한다.

```
from kubernetes import client
from kfserving import KFServingClient
from kfserving import constants
from kfserving import utils
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2TensorflowSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kubernetes.client import V1ResourceRequirements
import os
import sys
import argparse
import logging
import time

'''
'''
class KFServing(object):
    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--namespace', required=False, default='practice')
        # pvc://${PVCNAME}/dir
        parser.add_argument('--storage_uri', required=False, default='/saved_model')
        parser.add_argument('--name', required=False, default='kfserving-sample')        
        args = parser.parse_args()
        namespace = args.namespace
        serving_name =  args.name
        
        api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
        default_endpoint_spec = V1alpha2EndpointSpec(
                                  predictor=V1alpha2PredictorSpec(
                                    tensorflow=V1alpha2TensorflowSpec(
                                      storage_uri=args.storage_uri,
                                      resources=V1ResourceRequirements(
                                          requests={'cpu':'100m','memory':'1Gi'},
                                          limits={'cpu':'100m', 'memory':'1Gi'}))))
        isvc = V1alpha2InferenceService(api_version=api_version,
                                  kind=constants.KFSERVING_KIND,
                                  metadata=client.V1ObjectMeta(
                                      name=serving_name, namespace=namespace),
                                  spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))        
        
        KFServing = KFServingClient()
        KFServing.create(isvc)
        print('waiting 5 sec for Creating InferenceService')
        time.sleep(5)
        
        KFServing.get(serving_name, namespace=namespace, watch=True, timeout_seconds=300)
        
if __name__ == '__main__':
    if os.getenv('FAIRING_RUNTIME', None) is None:
        from kubeflow.fairing.builders.append.append import AppendBuilder
        from kubeflow.fairing.preprocessors.converted_notebook import \
            ConvertNotebookPreprocessor

        DOCKER_REGISTRY = 'ykkim77'
        base_image = 'brightfly/kubeflow-kfserving:latest'
        image_name = 'kfserving'

        builder = AppendBuilder(
            registry=DOCKER_REGISTRY,
            image_name=image_name,
            base_image=base_image,
            push=True,
            preprocessor=ConvertNotebookPreprocessor(
                notebook_file="serving.ipynb"
            )
        )
        builder.build()
    else:
        serving = KFServing()
        serving.run()
```



## pipeline 코드 작성하기

```
import kfp
import kfp.dsl as dsl
import kfp.onprem as onprem
import kfp.components as comp


    
def echo_op(text):
    return dsl.ContainerOp(
        name='echo',
        image='library/bash:4.4.23',
        command=['sh', '-c'],
        arguments=['echo "$0"', text],
    )  

@dsl.pipeline(
    name='FMnistPipeline',
    description='mnist '
)
def fmnist_pipeline(learning_rate, dropout_rate, epoch, act, layer,  
                    checkpoint_dir, saved_model_dir, pvc_name, tensorboard_log,
                    name, model_version, namespace):
  
    exit_task = echo_op("Done!")
    
    with dsl.ExitHandler(exit_task): 

        kubeflow_pvc = dsl.PipelineVolume(pvc=str(pvc_name))
        
        mnist = dsl.ContainerOp(
            name='FMnist',
            image='ykkim77/fairing-job:38041ACE',
            command=['python', '/app/fmniest_e2e.py'],
            arguments=[
                "--learning_rate", learning_rate,
                "--dropout_rate", dropout_rate,
                "--epoch", epoch,
                "--act", act,
                "--layer", layer,
                "--checkpoint_dir", checkpoint_dir,
                "--saved_model_dir", saved_model_dir,
                "--model_version", model_version,
                "--tensorboard_log", tensorboard_log
            ],
            pvolumes={"/result": kubeflow_pvc}
        )
        
        result = dsl.ContainerOp(
            name='list_list',
            image='library/bash:4.4.23',
            command=['ls', '-R', '/result'],
            pvolumes={"/result": mnist.pvolume}
        )
        
        kfserving = dsl.ContainerOp(
            name='kfserving',
            image='ykkim77/kfserving:B897A99B',
            command=['python', '/app/serving.py'],
            arguments=[
                "--namespace", namespace,
                "--storage_uri", "pvc://" +  str(pvc_name) + "/saved_model",
                "--name", name
            ]
        )        

        
        inference = dsl.ContainerOp(
            name='inference',
            image='library/bash:4.4.23',
            command=['curl -v -H  "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}/v1/models/kfserving-fmnist:predict -d @./input.json']
        )  
        
        result.after(mnist)
        kfserving.after(result)
        inference.after(kfserving)
        
    

arguments = {'learning_rate': '0.001397',
             'dropout_rate': '0.18',
             'epoch' : '11',
             'act' : 'sigmoid',
             'layer': '2',
             'checkpoint_dir': '/reuslt/training_checkpoints',
             'saved_model_dir':'/result/saved_model/',
             'pvc_name' : 'efs-claim',
             'tensorboard_log': '/result/log',
             'name' : 'kfserving-fmnist',
             'model_version' : '0001',
             'namespace' : 'practice'
            }
    
if __name__ == '__main__':
#     kfp.Client().create_run_from_pipeline_func(pipeline_func=fmnist_pipeline, 
#                                                arguments=arguments)
    kfp.compiler.Compiler().compile(fmnist_pipeline, 'fmnist.pipeline.tar.gz')
```
