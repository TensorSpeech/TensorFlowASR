# %% Imports
import os

import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()


# %% Load model

config_path = f"{os.path.dirname(__file__)}/../../../models/wordpiece-conformer-v2/config.yml"

config = Config(config_path)
tf.random.set_seed(0)
tf.keras.backend.clear_session()

speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(config=config)

h5 = f"{os.path.dirname(__file__)}/../../../models/wordpiece-conformer-v2/21.h5"

# build model
conformer = Conformer(**config.model_config, vocab_size=text_featurizer.num_classes)
conformer.make(speech_featurizer.shape)
conformer.add_featurizers(speech_featurizer, text_featurizer)
conformer.summary()
# conformer.load_weights(h5, by_name=True)

# %% Gen bp

output_bp = f"{os.path.dirname(__file__)}/../../../models/wordpiece-conformer-v2/model_only_bp"

tf.saved_model.save(conformer, output_bp)

# %% Load bp
loaded_conformer = tf.saved_model.load(output_bp)

# %%
