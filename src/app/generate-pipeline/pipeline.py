import kfp
from kfp import dsl

def step1():
    vop = dsl.VolumeOp(
        name="pvc",
        resource_name="pvc",
        size ='1Gi', 
        modes = dsl.VOLUME_MODE_RWO
    )
    return dsl.ContainerOp(
        name = 'preprocessing',
        image = 'minio999/preprocessing:v1.1',
        command = ['python3', 'preprocessing.py'],
        # pvolumes={
        #     '/app': vop.volume
        # }
    )

def step2(step1):
    return dsl.ContainerOp(
        name = 'Train',
        image = 'minio999/train:v1',
        command = ['python3', 'train.py'],
        # pvolumes = {
        #     '/app': vop.volume
        # },
    )

def step3(step2):
    return dsl.ContainerOp(
        name = 'Eval',
        image = 'minio999/eval:v1',
        command = ['python3', 'eval.py'],
        # pvolumes = {
        #     '/app': vop.volume
        # }
    )

@dsl.pipeline(
    name = 'titanic pipeline',
    description = 'Pipeline to detect if somebody would survived titanic crash')
def pipeline():
    preprocessing = step1()
    train = step2(preprocessing)
    eval = step3(train)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(pipeline, __file__ + '.yaml')