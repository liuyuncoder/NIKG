from logging import getLogger
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.utils import init_logger, init_seed, set_color, get_environment
from recbole.model.knowledge_aware_recommender.kgat import KGAT
# from recbole.model.knowledge_aware_recommender.kgcn import KGCN
import torch.distributed as dist

from recbole.utils import get_trainer
from model_KGAT import NIKG
# from model_KGCN import NIKG
import torch.multiprocessing as mp
from collections.abc import MutableMapping

def run_recbole(model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None):
    # kg_model = KGCN
    
    # configurations initialization
    config = Config(model=model, dataset=dataset,
                    config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = NIKG(config, train_data._dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process

def run_recboles(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole(
        *args[:3],
        **kwargs
    )

if __name__ == '__main__':

    kg_model = KGAT
    config_file_list = [
            '/home/yun-liu/workspace/NIKG/config/data.yaml', '/home/yun-liu/workspace/NIKG/config/model.yaml']
    args = dict(
        model = kg_model,
        dataset = 'amazon-books',
        config_file_list = config_file_list,
        world_size = 2,
        ip='localhost',
        port='5678',
        nproc = 2,
        group_offset = 0
    )

    # Optional, only needed if you want to get the result of each process.
    queue = mp.get_context('spawn').SimpleQueue()
    config_dict = None or {}
    config_dict.update({
        "world_size": args['world_size'],
        "nproc": args['nproc'],
        "ip": args['ip'],
        "port": args['port'],
        "offset": args['group_offset']
    })

    kwargs = {
        "config_dict": config_dict,
        "queue": queue, # Optional
    }

    mp.spawn(
        run_recboles,
        args=(args['model'], args['dataset'], args['config_file_list'], kwargs),
        nprocs=args['nproc'],
        join=True
    )

    # Normally, there should be only one item in the queue
    res = None if queue.empty() else queue.get()