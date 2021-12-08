import torch
import functools
from absl import flags
from absl import app
from oatomobile.benchmarks.carnovel.benchmark import carnovel
from oatomobile.baselines.torch.cim.model import ImitativeModel
from oatomobile.baselines.torch.cim.agent import CIMAgent
from oatomobile.baselines.torch.cim.predictor.model import MLP

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="task",
    default=None,
    help="Name of the task.",
)
flags.DEFINE_string(
    name="model_dir",
    default=None,
    help="Path of the imitative model.",
)
flags.DEFINE_string(
    name="predictor_dir",
    default=None,
    help="Path of the speed predictor.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="Path to output",
)
flags.DEFINE_float(
    name="alpha",
    default=0.1,
    help="Value of alpha.",
)
flags.DEFINE_float(
    name="gamma",
    default=0.1,
    help="Value of gamma.",
)
flags.DEFINE_integer(
    name="gpu",
    default=0,
    help="GPU id",
)

def main(argv):
    task = FLAGS.task
    model_dir = FLAGS.model_dir
    predictor_dir = FLAGS.predictor_dir
    output_dir = FLAGS.output_dir
    alpha = FLAGS.alpha
    gamma = FLAGS.gamma
    gpu = FLAGS.gpu
    weather = 'ClearNoon'

    model = ImitativeModel()
    model = torch.load(model_dir)
    model = model.cuda(gpu)
    speedmodule = MLP(lags = 3)
    speedmodule = torch.load(predictor_dir)
    speedmodule = speedmodule.cuda(gpu)
    agent_fn = functools.partial(CIMAgent, model=model, speedmodule=speedmodule, lags = 3, horizon = 10, alpha = alpha, gamma = gamma)
    return carnovel.evaluate(agent_fn, log_dir=output_dir, subtasks_id=task, monitor=False, render=False, weather=weather)

if __name__ == "__main__":
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("predictor_dir")
    app.run(main)