# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv
from src.data.extract_card import extract_cards_from_video
import os
from pathlib import Path
import shutil


@click.command()
@click.argument(
    'input_path',
    type=click.Path(exists=True),
    default="data/raw/video"
)
@click.argument(
    'output_path',
    type=click.Path(),
    default="data/processed/cards"
)
@click.argument(
    'input_file_extension',
    type=click.STRING,
    default="mp4"
)
@click.argument(
    'card_suits',
    default=','.join(['s', 'h', 'd', 'c']),
    type=click.STRING
)
@click.argument(
    'card_values',
    default=','.join(['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6']),
    type=click.STRING
)
def make_dataset(
    input_path: str = "data/raw/video",
    output_path: str = "data/processed/cards",
    input_file_extension: str = "mp4",
    card_suits: str = 's,h,d,c',
    card_values: str = 'A,K,Q,J,10,9,8,7,6'
):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('-----------------------------------------------------------')
    logger.info('Extracting card values under different lighting conditions.')
    logger.info('-----------------------------------------------------------')

    card_suits = [c.strip() for c in card_suits.split(',')]
    card_values = [c.strip() for c in card_values.split(',')]

    output_path = Path(output_path)

    if output_path.exists():
        shutil.rmtree(output_path)

    for suit in card_suits:
        for value in card_values:

            card_name = value+suit
            video_filename = os.path.join(
                input_path, card_name+"."+input_file_extension)

            card_path = Path(os.path.join(output_path, card_name))
            os.makedirs(card_path)

            imgs = extract_cards_from_video(video_filename, card_path)
            logger.info(f"Extracted images for {card_name} : {len(imgs)}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_dataset()
