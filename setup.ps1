# local instalation of kubeflow via documentation https://www.kubeflow.org/docs/components/pipelines/v1/installation/localcluster-deployment/
export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# connection via http://localhost:8080/
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# NOTE: setuping seldon core via documention
# Portforword seldon deployment 
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80