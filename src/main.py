from logging import getLogger
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.utils import init_logger, init_seed, set_color
# from recbole.model.knowledge_aware_recommender.kgat import KGAT
from recbole.model.knowledge_aware_recommender.kgcn import KGCN
import torch.distributed as dist

from recbole.utils import get_trainer
# from model_KGAT import NIKG
from model_KGCN import NIKG
# import torch.multiprocessing as mp
# from collections.abc import MutableMapping

def run(model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True):
    # kg_model = KGCN
    
    # configurations initialization
    config = Config(model=model, dataset='amazon-books',
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

    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    count = 0
    info = '\nbest valid score:'
    for i in best_valid_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % 3
        info += "{:15}:{:<10}    ".format(i, best_valid_result[i])
    logger.info(info)

    count = 0
    info = '\ntest result:'
    for i in test_result.keys():
        if count == 0:
            info += '\n'
        count = (count + 1) % 3
        info += "{:15}:{:<10}    ".format(i, test_result[i])
    logger.info(info)

if __name__ == '__main__':

    kg_model = KGCN
    run(model=kg_model, config_file_list=[
            '/home/yun-liu/workspace/NIKG/config/data.yaml', '/home/yun-liu/workspace/NIKG/config/model.yaml'])
    