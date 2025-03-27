import copy
from typing import List
from c2m3.modules.pl_module import MyLightningModule
from c2m3.match.permutation_spec import CNNPermutationSpecBuilder
from c2m3.match.merger import FrankWolfeSynchronizedMerger, FrankWolfeToReferenceMerger
from c2m3.match.utils import restore_original_weights

def frank_wolfe_synchronized_merging(models: List[MyLightningModule], train_loaders):

    symbols_to_models = {}
    for index, model in enumerate(models):
        # a, b, c...
        symbols_to_models[chr(97 + index)] = model
    
    print(symbols_to_models.keys())
    print(type(list(symbols_to_models.values())[0]))

    # extract first model
    ref_model = copy.deepcopy(symbols_to_models[list(symbols_to_models.keys())[0]])

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in symbols_to_models.items()}

    permutation_spec_builder = CNNPermutationSpecBuilder()
    permutation_spec = permutation_spec_builder.create_permutation_spec(ref_model=ref_model)

    restore_original_weights(symbols_to_models, model_orig_weights)


    # model_merger = FrankWolfeToReferenceMerger(
    #     name="FrankWolfeSync", 
    #     permutation_spec=permutation_spec,
    #     initialization_method="identity" #identity, random, sinkhorn, LAP, bistochastic_barycenter
    #     )
    model_merger = FrankWolfeSynchronizedMerger(
        name="FrankWolfeSync", 
        permutation_spec=permutation_spec,
        initialization_method="identity" #identity, random, sinkhorn, LAP, bistochastic_barycenter
        )
    merged_model, repaired_model, models_permuted_to_universe = model_merger(symbols_to_models, train_loader=train_loaders)

    print(f"Successfully merged models.")

    return repaired_model

