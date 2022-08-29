import argparse
import numpy as np

from util import write_w2v, writeAnalogies, writeGroupAnalogies, convert_legacy_to_keyvec, load_legacy_w2v, pruneWordVecs
from biasOps import identify_bias_subspace, neutralize_and_equalize, equalize_and_soften, normalize
from evalBias import generateAnalogies, multiclass_evaluation
from loader import load_def_sets, load_analogy_templates, load_test_terms, load_eval_terms
from scipy.stats import ttest_rel, spearmanr
from polysemy import *
from visualize import *

parser = argparse.ArgumentParser()
parser.add_argument('-embeddingPath', default='data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v_0.w2v')
parser.add_argument('vocabPath', choices=['inter', 'gender', 'race', 'religion', 'grr', 'gr'])
parser.add_argument('subs', choices=['default', 'josec', 'sum', 'mean', 'concat'])
parser.add_argument('eval', choices=['inter', 'gender', 'race', 'religion', 'total'])
parser.add_argument('-hard', action='store_true')
parser.add_argument('-soft', action='store_true')
parser.add_argument('-w', action='store_true')
parser.add_argument('-v', action='store_true')
parser.add_argument('-k', type=int, default=2)
parser.add_argument('-g', action='store_true') #visualize
parser.add_argument('-printLimit', type=int, default=500)
parser.add_argument('-analogies', action="store_true")
args = parser.parse_args()

outprefix = args.vocabPath + "_" + args.subs
embprefix = args.embeddingPath.replace("/", "_").replace("\\", "_").replace(".", "_").replace("attributes_", "").replace("data_", "").replace("json_", "").replace("w2vs_", "").replace("_w2v", "").replace("output_", "")

print("Loading embeddings from {}".format(args.embeddingPath))
word_vectors, embedding_dim = load_legacy_w2v(args.embeddingPath)

# Debiasing Subspace
if(args.vocabPath=='inter'):
    path = 'data/vocab/inter_attributes_optm.json'
    mode = 'attribute'
elif(args.vocabPath=='gender'):
    path = 'data/vocab/gender_attributes_optm.json'
    mode = 'role'
elif (args.vocabPath == 'race'):
    path = 'data/vocab/race_attributes_optm.json'
    mode = 'role'
elif(args.vocabPath=='religion'):
    path = 'data/vocab/religion_attributes_optm.json'
    mode = 'attribute'
elif (args.vocabPath == 'grr'):
    path = 'data/vocab/genderracereligion_optm.json'
    mode = 'role'
elif (args.vocabPath == 'gr'):
    path = 'data/vocab/genderrace_optm.json'
    mode = 'role'
else:
    raise Exception("Invalid path")

# EvalSet Setting
if(args.eval == 'inter'):
    print("Evaluation Set: Intersection")
    evalTargets, evalAttrs = load_eval_terms('data/vocab/inter_attributes_optm.json', 'attribute')
elif(args.eval == 'gender'):
    print("Evaluation Set: Gender")
    evalTargets, evalAttrs = load_eval_terms('data/vocab/gender_attributes_optm.json', 'role')

elif(args.eval == 'race'):
    print("Evaluation Set: Race")
    evalTargets, evalAttrs = load_eval_terms('data/vocab/race_attributes_optm.json', 'role')

elif(args.eval == 'religion'):
    print("Evaluation Set: Religion")
    evalTargets, evalAttrs = load_eval_terms('data/vocab/religion_attributes_optm.json', 'attribute')

elif(args.eval == 'total'):
    print("Evaluation Set: Total")
    evalTargets, evalAttrs = load_eval_terms('data/vocab/genderracereligion_optm.json', 'role')

else:
    raise Exception("Invalid path")

print("Loading vocabulary from {}".format(path))

analogyTemplates = load_analogy_templates(path, mode)
defSets = load_def_sets(path)

# #########NY: set debiasing word set to all given gender-race-religion words
# analogyTemplates = load_analogy_templates('data/vocab/genderracereligion_optm.json', 'role')
# defSets = load_def_sets('data/vocab/genderracereligion_optm.json')

testTerms = load_test_terms(path)

neutral_words = []
for value in analogyTemplates.values():
    neutral_words.extend(value)


print("Pruning Word Vectors... Starting with", len(word_vectors))
word_vectors = pruneWordVecs(word_vectors)
print("\tEnded with", len(word_vectors))

print("Identifying bias subspace")
subspace = identify_bias_subspace(word_vectors, defSets, args.k, embedding_dim)[:args.k]
genDefSets = load_def_sets('data/vocab/gender_attributes_optm.json')
racDefSets = load_def_sets('data/vocab/race_attributes_optm.json')
relDefSets = load_def_sets('data/vocab/religion_attributes_optm.json')

if(args.subs == 'default'):
    final_subspace = subspace

elif(args.subs =='josec'):
    subspace_gen = identify_bias_subspace(word_vectors, genDefSets, args.k, embedding_dim)[:args.k]
    subspace_rac = identify_bias_subspace(word_vectors, racDefSets, args.k, embedding_dim)[:args.k]
    subspace_rel = identify_bias_subspace(word_vectors, relDefSets, args.k, embedding_dim)[:args.k]
    contextVecs = np.array([subspace_gen, subspace_rac, subspace_rel])
    K = 1
    senNum, rank, dim = contextVecs.shape  # (3, 2, 50)
    kmeansIterMax = 1000

    final_subspace = polysemy(contextVecs, K, 50, kmeansIterMax, senNum)  # (1, 50)
    print("Applying Polysemy")
    # np.savetxt("data/gen_vector.csv", subspace_gen, delimiter=",")
    # np.savetxt("data/rac_vector.csv", subspace_rac, delimiter=",")
    # np.savetxt("data/rel_vector.csv", subspace_rel, delimiter=",")
    # np.savetxt("data/josec_vector.csv", final_subspace, delimiter=",")


# make sure use 'genderracereligion_optm.json' for vocabPath
else:
    subspace_gen = identify_bias_subspace(word_vectors, genDefSets, args.k, embedding_dim)[:args.k]
    subspace_rac = identify_bias_subspace(word_vectors, racDefSets, args.k, embedding_dim)[:args.k]
    subspace_rel = identify_bias_subspace(word_vectors, relDefSets, args.k, embedding_dim)[:args.k]
    if(args.subs =='sum'):
        final_subspace = subspace_gen + subspace_rac + subspace_rel
    elif(args.subs == 'mean'):
        final_subspace = (subspace_gen + subspace_rac + subspace_rel) / 3
    elif(args.subs == 'concat'):
        final_subspace = np.concatenate((subspace_gen, subspace_rac, subspace_rel), axis=0)  # (6, 50)
    else:
        raise Exception("Please enter the type of the debiasing subspace")



if(args.hard):
    print("Neutralizing and Equalizing")
    new_hard_word_vectors = neutralize_and_equalize(word_vectors, neutral_words,
                        defSets.values(), final_subspace, embedding_dim)
if(args.soft):
    print("Equalizing and Softening")
    new_soft_word_vectors = equalize_and_soften(word_vectors, neutral_words,
                        defSets.values(), final_subspace, embedding_dim, verbose=args.v)

if(args.analogies):
    print("Generating Analogies")
    biasedAnalogies, biasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(word_vectors))
    if(args.hard):
        hardDebiasedAnalogies, hardDebiasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(new_hard_word_vectors))
    if(args.soft):
        softDebiasedAnalogies, softDebiasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(new_soft_word_vectors))

    if(args.w):
        print("Writing biased analogies to disk")
        writeAnalogies(biasedAnalogies, "output/" + outprefix + "_biasedAnalogiesOut.csv")
        writeGroupAnalogies(biasedAnalogyGroups, "output/" + outprefix + "_biasedAnalogiesOut_grouped.csv")

    if(args.v):
        print("Biased Analogies (0-" + str(args.printLimit) + ")")
        for score, analogy, _ in biasedAnalogies[:args.printLimit]:
            print(score, analogy)

    if(args.w):
        if(args.hard):
            print("Writing hard debiased analogies to disk")
            writeAnalogies(hardDebiasedAnalogies, "output/" + outprefix + "_hardDebiasedAnalogiesOut.csv")
            writeGroupAnalogies(hardDebiasedAnalogyGroups, "output/" + outprefix + "_hardDebiasedAnalogiesOut_grouped.csv")
        if(args.soft):
            print("Writing soft debiased analogies to disk")
            writeAnalogies(softDebiasedAnalogies, "output/" + outprefix + "_softDebiasedAnalogiesOut.csv")
            writeGroupAnalogies(softDebiasedAnalogyGroups, "output/" + outprefix + "_softDebiasedAnalogiesOut_grouped.csv")
    if(args.v):
        if(args.hard):
            print("="*20, "\n\n")
            print("Hard Debiased Analogies (0-" + str(args.printLimit) + ")")
            for score, analogy, _ in hardDebiasedAnalogies[:args.printLimit]:
                print(score, analogy)
        if(args.soft):
            print("="*20, "\n\n")
            print("Soft Debiased Analogies (0-" + str(args.printLimit) + ")")
            for score, analogy, _ in softDebiasedAnalogies[:args.printLimit]:
                print(score, analogy)
        
if(args.w):
    print("Writing data to disk")
    write_w2v("output/religion_evalset/poly/" + outprefix + "_biasedEmbeddingsOut.w2v", word_vectors)
    if(args.hard):
        write_w2v("output/religion_evalset/poly/" + outprefix + "_hardDebiasedEmbeddingsOut.w2v", new_hard_word_vectors)
    if(args.soft):
        write_w2v("output/" + embprefix + "_" + outprefix + "_softDebiasedEmbeddingsOut.w2v", new_soft_word_vectors)

print("--Performing Evaluation")


print("--Biased Evaluation Results")
biasedMAC, biasedDistribution = multiclass_evaluation(word_vectors, evalTargets, evalAttrs)
print("Biased MAC:", biasedMAC)

if(args.hard):
    print("--HARD Debiased Evaluation Results")
    debiasedMAC, debiasedDistribution = multiclass_evaluation(new_hard_word_vectors, evalTargets, evalAttrs)
    print("HARD MAC:", debiasedMAC)

    statistics, pvalue = ttest_rel(biasedDistribution, debiasedDistribution)
    print("--HARD Debiased Cosine difference t-test", pvalue)

if(args.soft):
    print("SOFT Debiased Evaluation Results")
    debiasedMAC, debiasedDistribution = multiclass_evaluation(new_soft_word_vectors, evalTargets, evalAttrs)
    print("SOFT MAC:", debiasedMAC)

    statistics, pvalue = ttest_rel(biasedDistribution, debiasedDistribution)
    print("SOFT Debiased Cosine difference t-test", pvalue)

if(args.w):
    print("--Writing statistics to disk")
    f = open("output/" + outprefix + "_statistics.csv", "w")
    f.write("Biased MAC,Debiased MAC,P-Value\n")
    f.write(str(biasedMAC) + "," +  str(debiasedMAC) + "," + str(pvalue) + "\n")
    f.close()

if(args.g):
    # make x-y axis vector
    diff_gen = bias_diff_vector(word_vectors, genDefSets, genDefSets) # he-she
    diff_rac = bias_diff_vector(word_vectors, racDefSets, racDefSets) # black-white
    norm_word_vectors = normalize(word_vectors)

    load_attributes = load_analogy_templates('data/vocab/inter_attributes_optm.json', 'attribute')
    load_random = load_analogy_templates('data/vocab/inter_attributes_optm.json', 'random')
    af_f_attr = load_attributes["af_f"]
    af_m_attr = load_attributes["af_m"]
    eu_f_attr = load_attributes["eu_f"]
    eu_m_attr = load_attributes["eu_m"]
    random_words = load_random["insects"]

    af_f_biased, af_f_label = label(word_vectors, af_f_attr)
    af_m_biased, af_m_label = label(word_vectors, af_m_attr)
    eu_f_biased, eu_f_label = label(word_vectors, eu_f_attr)
    eu_m_biased, eu_m_label = label(word_vectors, eu_m_attr)
    random_biased, random_label = label(word_vectors, random_words)

    af_f_debiased, _ = label(new_hard_word_vectors, af_f_attr)
    af_m_debiased, _ = label(new_hard_word_vectors, af_m_attr)
    eu_f_debiased, _ = label(new_hard_word_vectors, eu_f_attr)
    eu_m_debiased, _ = label(new_hard_word_vectors, eu_m_attr)
    random_debiased, _ = label(new_hard_word_vectors, random_words)

    color_af_f = color("yellow", af_f_label)
    color_af_m = color("blue", af_m_label)

    color_eu_f = color("m", eu_f_label)
    color_eu_m = color("c", eu_m_label)

    color_r = color("black", random_label)
    color_target = ["red"]

    # # Before Debiasing
    # title = "Biased Attribute Words"
    # new_visualize(np.vstack((subspace, af_f_biased, af_m_biased, eu_f_biased, eu_m_biased)), diff_gen, diff_rac,
    #             ["target"] + af_f_label + af_m_label + eu_f_label + eu_m_label,
    #             color_target + color_af_f + color_af_m + color_eu_f + color_eu_m,
    #             "{} on Gender/Race WO Random Words".format(title))
    #
    # # Without random words
    # title = "Debiased Attribute Words"
    # new_visualize(np.vstack((subspace, af_f_debiased, af_m_debiased, eu_f_debiased, eu_m_debiased)), diff_gen, diff_rac,
    #               ["target"] + af_f_label + af_m_label + eu_f_label + eu_m_label,
    #               color_target + color_af_f + color_af_m + color_eu_f + color_eu_m,
    #               "{} on Gender/Race WO Random Words".format(title))
    #
    # # With random words
    # new_visualize(np.vstack((subspace, af_f_debiased, af_m_debiased, eu_f_debiased, eu_m_debiased, random)), diff_gen, diff_rac,
    #               ["target"] + af_f_label + af_m_label + eu_f_label + eu_m_label + random_label,
    #               color_target + color_af_f + color_af_m + color_eu_f + color_eu_m + color_r,
    #               "{} - on Gender/Race W Random Words".format(title))

    title = "Biased Intersectional Attribute Words"
    visualize(np.vstack((subspace, af_f_biased, af_m_biased, eu_f_biased, eu_m_biased)),
                  ["target"] + af_f_label + af_m_label + eu_f_label + eu_m_label,
                  color_target + color_af_f + color_af_m + color_eu_f + color_eu_m,
                  "{} WO Random Words".format(title))

    title = "Debiased Intersectional Attribute Words"
    visualize(np.vstack((subspace, af_f_debiased, af_m_debiased, eu_f_debiased, eu_m_debiased)),
              ["target"] + af_f_label + af_m_label + eu_f_label + eu_m_label,
              color_target + color_af_f + color_af_m + color_eu_f + color_eu_m,
              "{} WO Random Words".format(title))