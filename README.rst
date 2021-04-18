Explainability-Based Attention Head Analysis for Transformers
==============================================================

Notebook with examples of visualizations:
----------------------------

|examples|

.. |examples| image:: https://colab.research.google.com/assets/colab-badge.svg
                   :target: https://colab.research.google.com/github/hila-chefer/NLP_Final_Project/blob/main/Explainability_Based_Attention_Head_Analysis_for_Transformers.ipynb

.. sectnum::

Using Colab
----------------

* Installing all the requirements may take some time. After installation, please restart the runtime.

Reproduction of results
-----------------------
^^^^^^^^^^^^^^^^^^^^
Downloading the data
^^^^^^^^^^^^^^^^^^^^
Our VisualBERT implementation uses the mmf library, which will automatically download all the data you need for all the experiments, once you run one of the VisualBERT experiments. The path to the data will be the one you set in the environment variable ``env.data_dir``. 

In the path noted in ``env.data_dir``, you will have the COCO 2014 validation dataset in path: ``path_to_data_dir/datasets/coco/subset_val/images/val2014/``. This is the value that should be used as the COCO_path environment variable in all your experiments.
For example, if the path in ``env.data_dir`` was ``/home`` then the COCO_path environment variable in all the experiments should be: ``/home/datasets/coco/subset_val/images/val2014/``

^^^^^^^^^^
VisualBERT
^^^^^^^^^^

Run the ``run.py`` script as follows:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python run.py config=projects/visual_bert/configs/vqa2/defaults.yaml model=visual_bert dataset=vqa2 run_type=val checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train env.data_dir=/path/to/data_dir training.num_workers=0 training.batch_size=1 training.trainer=<mmf_head_prun/mmf_layer_prun/mmf_exp> training.seed=1234 prune.num_of_examples=10000 prune.is_grad=False prune.positive_prune=False prune.COCO_path=/path/to/data_dir/datasets/coco/subset_val/images/val2014/

.. note::

 * If you already downloaded the data, ``env.data_dir`` should be the path to the downloaded data. The ``env.data_dir`` directory is expected to contain the COCO 2014 validation data. If the datasets aren't already in ``env.data_dir``, then the script will download the data automatically to the path in ``env.data_dir``.
 * ``training.trainer`` should be one of: "mmf_head_prun", "mmf_layer_prun", "mmf_exp" depending on whether you want to prune heads/layers or generate visualizations.
 * To run using our scoring method use ``prune.is_grad=True``. To use LRP run ``prune.is_grad=False``.
 * When pruning heads/layers, to prune largest first (what we call *positive*) use ``prune.positive_prune=True``. Otherwise use ``prune.positive_prune=False``.
  


^^^^^^
LXMERT
^^^^^^

#. Download `valid.json <https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json>`_:

    .. code-block:: bash

      pushd data/vqa
      wget https://nlp.cs.unc.edu/data/lxmert_data/vqa/valid.json
      popd

#. Download the `COCO_val2014` set to your local machine.

   .. note::

      If you already downloaded `COCO_val2014` for the `VisualBERT`_ tests, you can simply use the same path you used for `VisualBERT`_.

#. Run the ``perturbation.py`` script as follows:

    .. code-block:: bash

      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python lxmert/lxmert/perturbation.py  --COCO_path /path/to/data_dir/datasets/coco/subset_val/images/val2014/ --method <ours/lrp> --is-positive-pert True --num-samples 10000 --prune_type <head/layer> --seed 1234
      
   .. note::

      * ``method=ours`` will run our method for scoring heads. To use LRP set ``method=lrp``.
      * To run negative pruning (where less important heads are removed first) remove the flag ``--is-positive-pert True``.
      * ``COCO_path`` should be the same path used in the VisualBERT experiments, i.e. ``/path/to/data_dir/datasets/coco/subset_val/images/val2014/``.
      * ``prune_type`` should be ``head/layer`` depending on whether you want to prune heads/layers.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generating Visualizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**VisualBERT:**
  Run the ``run.py`` script as follows:
.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python run.py config=projects/visual_bert/configs/vqa2/defaults.yaml model=visual_bert dataset=vqa2 run_type=val checkpoint.resume_zoo=visual_bert.finetuned.vqa2.from_coco_train env.data_dir=/path/to/data_dir training.num_workers=0 training.batch_size=1 training.trainer=mmf_exp training.seed=1 prune.num_of_examples=2 prune.is_grad=False prune.positive_prune=False prune.COCO_path=/path/to/data_dir/datasets/coco/subset_val/images/val2014/

.. note::

* This will run visualizations for pruning 0%, 40%, 60%, 90% of the largest heads using our method.
* Results are saved to the root project directory.
* The example shown in the paper is the second one.

**LXMERT:**
  Run the ``generate_visualization.py`` script as follows:
.. code-block:: bash

      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=`pwd` python lxmert/lxmert/generate_visualization.py  --COCO_path /path/to/data_dir/datasets/coco/subset_val/images/val2014/ --num-samples 2 --seed 1234
 
.. note::

* This will run visualizations for pruning 0%, 40%, 60%, 90% of the least significant heads using our method.
* Results are saved to the root project directory.
* The example shown in the paper is the second one.
Code
----

Since we use MMF as a base, which is a large repository, we attach a list of files we have changed/added:

* /run.py
* /mmf/models/transformers/backends/BERT_ours.py
* /mmf/models/transformers/backends/ExplanationGenerator.py
* /mmf/models/transformers/backends/VisualizationGenerator.py
* /mmf/trainers/mmf_trainer.exp
* /mmf/trainers/core/evaluation_loop
* /lxmert/lxmert/generate_visualization.py
* /lxmert/lxmert/perturbation.py
* /lxmert/lxmert/src/ExplanationGenerator.py
* /lxmert/lxmert/src/VisualizationGenerator.py
* /lxmert/lxmert/src/param.py
* /lxmert/lxmert/src/huggingface_lxmert.py
Credits
-------

* VisualBERT implementation is based on the `MMF <https://github.com/facebookresearch/mmf>`_ framework.
* LXMERT implementation is based on the `offical LXMERT <https://github.com/airsplay/lxmert>`_ implementation and on `Hugging Face Transformers <https://github.com/huggingface/transformers>`_.

