import numpy as np
import torch
import pandas as pd
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class OLMWordSwap(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.
    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """
    index = 0
    def __init__(self, wir_method, path_to_olm_csv, path_to_examples_csv):
        self.path_to_olm_csv = path_to_olm_csv
        self.path_to_examples_csv = path_to_examples_csv
        self.wir_method = wir_method

    def _process_the_csv(self, df, index_of_mapping):
        tokens_for_mapping_with_TA = eval(df['tokenized_text'].iloc[index_of_mapping])
        token_scores_for_mapping_with_TA = eval(df['relevances'].iloc[index_of_mapping])
        
        # list_scores_in_string_format = df["relevances"].to_list()

        # tokens_for_mapping_with_TA = []
        # token_scores_for_mapping_with_TA = []

        # for i,pt in enumerate(df["tokenized_text"].to_list()):
        #     pt = pt[1:-1]
        #     l = pt.split(", ")
        #     l = [a[1:-1] for a in l]

        #     sc = list_scores_in_string_format[i]
        #     sc = sc[1:-1]
        #     sclist = sc.split(", ")
        #     sclist = [float(_) for _ in sclist]

        #     tokens_for_mapping_with_TA.append(l)
        #     token_scores_for_mapping_with_TA.append(sclist)
            
        return tokens_for_mapping_with_TA, token_scores_for_mapping_with_TA

        
    # def _text_att_tokenize_with_lower(self, s, words_to_ignore=[]):
    #     s = s.replace("<br />"," ")
    #     words = []
    #     word = ""
    #     for c in " ".join(s.split()):
    #         if c.isalnum():
    #             word += c
    #         elif c in "'-" and len(word) > 0:
    #             # Allow apostrophes and hyphens as long as they don't begin the
    #             # word.
    #             word += c
    #         elif word:
    #             if word not in words_to_ignore:
    #                 words.append(word.lower())
    #             word = ""
    #     if len(word) and (word not in words_to_ignore):
    #         words.append(word.lower())

    #     return words


    # def _get_index(self, text_attack_toks, df_examples):
    #     text_attack_toks_lower = [t.lower() for t in text_attack_toks]
    #     text_attack_toks_shrinked = text_attack_toks_lower[:min(10, len(text_attack_toks_lower)-1)]
    #     list_of_examples = df_examples["sentence"].to_list()
        
    #     print(text_attack_toks_shrinked)

    #     tokenized_samples = [self._text_att_tokenize_with_lower(example) for example in list_of_examples]
    #     tokenized_samples_shrinked = [example_toks[:min(len(example_toks)-1, 10)] for example_toks in tokenized_samples]
        
    #     print(tokenized_samples_shrinked)
        
    #     index_to_return = []

    #     for i,sample in enumerate(tokenized_samples_shrinked):
    #         if(sample == text_attack_toks_shrinked):
    #             index_to_return.append(i)

    #     print(index_to_return)
    #     assert len(index_to_return) == 1
    #     return index_to_return[0]


    # def _get_mapped_scores_for_olm(self, token_scores_for_mapping_with_TA, tokens_for_mapping_with_TA, text_attack_toks):
    #     new_tok_scores = []  # holds token scores for elements in text attack toks 
    #     matches_toks = []    # holds token that are same in both 

    #     # for i, n_tok in enumerate(text_attack_toks):
    #     #     l_s_x = []
    #     #     l_s_y = []
    #     #     for j, tok in enumerate(tokens_for_mapping_with_TA):
    #     #         if(tok == n_tok):
    #     #             l_s_x.append(j)
    #     #             l_s_y.append(abs(i-j))

    #     #     if(len(l_s_x)==0):
    #     #         new_tok_scores.append(0)
    #     #         matches_toks.append('_')
    #     #         continue
            
    #     #     [l_s_x for _,x in sorted(zip(l_s_y,l_s_x))]
    #     #     #l_s = np.sort(l_s);
    #     #     #print(l_s)
    #     #     new_tok_scores.append(token_scores_for_mapping_with_TA[l_s_x[0]])
    #     #     matches_toks.append(tokens_for_mapping_with_TA[l_s_x[0]])

    #     # print(new_tok_scores)
    #     # print()
    #     # print(matches_toks)
    #     # if len(text_attack_toks) == len(tokens_for_mapping_with_TA):
    #     return token_scores_for_mapping_with_TA

    #     # elif len(text_attack_toks) < len(tokens_for_mapping_with_TA):
    #     #     ta_idx = 0
    #     #     first = False

    #     #     for idx, tok in enumerate(tokens_for_mapping_with_TA):
    #     #         if text_attack_toks[ta_idx] == tok:
    #     #             matches_toks.append(tok)
    #     #             new_tok_scores.append(token_scores_for_mapping_with_TA[idx])
    #     #             ta_idx += 1
    #     #             first = False
                
    #     #         elif first==False:
    #     #             matches_toks.append(tok + tokens_for_mapping_with_TA[idx+1])
    #     #             new_tok_scores.append(token_scores_for_mapping_with_TA[idx])
    #     #             first = True
    #     #             ta_idx += 1

    #     # if len(text_attack_toks) > len(tokens_for_mapping_with_TA):
    #     #     ta_idx = 0
    #     #     first = False

    #     #     for idx, tok in enumerate(text_attack_toks):
    #     #         if tokens_for_mapping_with_TA[ta_idx] == tok:
    #     #             matches_toks.append(tok)
    #     #             new_tok_scores.append(token_scores_for_mapping_with_TA[idx])
    #     #             ta_idx += 1
    #     #             first = False
                
    #     #         elif first==False:
    #     #             matches_toks.append(tok)
    #     #             new_tok_scores.append(token_scores_for_mapping_with_TA[idx])
    #     #             new_tok_scores.append(0)
    #     #             first = True
    #     #             ta_idx += 1
            

    #     print("TA TOkens: ", text_attack_toks)
    #     print("$$Matched tokens: ", matches_toks)
    #     print(len(text_attack_toks), len(matches_toks))

    #     return new_tok_scores

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        text_attack_toks = initial_text.words
        # print(f"$$$$$$$$$$$$$$$$$$$${initial_text}$$$$$$$$")
        text_attack_toks = [tok.lower() for tok in text_attack_toks]
        
        df_olm = pd.read_csv(self.path_to_olm_csv)
        
        index_of_mapping = OLMWordSwap.index
        OLMWordSwap.index += 1

        # ##############

        # print(index_of_mapping)
        # print(text_attack_toks)
        # olm_df = eval(df_olm['tokenized_text'].iloc[index_of_mapping])
        # print(olm_df)
        # print(len(text_attack_toks), len(olm_df))

        # #########

        # tokens_for_mapping_with_TA, token_scores_for_mapping_with_TA = self._process_the_csv(df_olm, index_of_mapping)
        # mapped_scores = self._get_mapped_scores_for_olm(
        #     token_scores_for_mapping_with_TA, tokens_for_mapping_with_TA, text_attack_toks
        # )
        
        mapped_scores = eval(df_olm['relevances'].iloc[index_of_mapping])
        # print(len(mapped_scores))
        # print(mapped_scores)
        index_scores = np.array(mapped_scores)
        search_over = False        

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over


    def _perform_search(self, initial_result):
    
        attacked_text = initial_result.attacked_text
        # print(attacked_text)
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result


    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
