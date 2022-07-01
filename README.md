# Katib 를 활용한 HPO 실습

## YAML 파일 기반의 Katib 수행 명세

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
    algorithmName: random   # HPO Method
  parallelTrialCount: 3       # 한번에 병렬화 할 수 있는 수
  maxTrialCount: 12          # 총 시도 횟수
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
                image: ykkim77/fairing-job:38041ACE       # HPO를 적용한 모델
                command:
                  - "python"
                  - "/app/fmniest_e2e.py"
                  - "--learning_rate=${trialParameters.learning_rate}"
                  - "--dropout_rate=${trialParameters.dropout_rate}"
            restartPolicy: Never
```
