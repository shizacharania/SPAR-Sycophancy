Explanation of Nina Rimsky's code on Sycophancy Steering:
(https://github.com/nrimsky/LM-exp/blob/main/sycophancy/sycophancy_steering.ipynb)

In [48]: take all the datasets from hugging face repository
 - https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_nlp_survey.jsonl
 - https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_philpapers2020.jsonl
 - https://huggingface.co/datasets/Anthropic/model-written-evals/raw/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl
In [81]: sort out all the data bsed on the question and answer matching th behviour (if u check out the links u can tell how they separate it)
In [84]: tqdm -> helps us in visualizing the completion of the loop. dhutil -> operations on a file
In [85]: will be referenced later

In [20]: AutoTokenizer.from_pretrained() is from Huggingface' transformers library. The purpose of this is to take the pretrained model and instantiate a tokenizer (a class responsible for converting text into tokens that can be fed into the model)
tokenizer.pad_token = tokenizer.eos_token
- pad_token = special token used for padding sequences to a consistent length when batching sequences together - this is easier so when u train stuff together its consistent and more efficient (used when training stuff in batches)
- with this statement, you're saying "use the same token for both padding and indicating the end of a sequence" and since blank lines are also valid story endings, it will treat them as end of story rather than meaningless filters

In [120]: from all the data, get 3000 samples/items

In [121]: make a class to compare sycophnatic and non sycophnatic test later on

In [123]: for all the layers in the model, you do layer.reset() -> self.model.model.layers[i] = BlockOutputWrapper(layer) -> this means that for every layer, you get the last hidden state values and it provides the capability to add certain activations to the output. but by resetting it, it just attribures the last_hidden_state and add_activations to none

In [125]: She only took activations from "intermediate activations at a late layer (I tested 28, 29, and 30)"
- filenames = dictionary of layer + making a filename with a timestamp along w the layer
- diffs = dictionary of each layer and [] for each layer
- the outer for loop goes through the s_tokens and n_tokens (sycophantic and non sycophantic) from the 3000 samples while using the tqdm feature to track the for loop progress + without any gradients when performing through the model -> then put into softmax = fio the probability of that word being the next one from the logits vector (also adds to 1). this gets you the logits
    - the first inner loop goes through all the sycophantic data layers to get last activations for the layer we are iterating over
        - s_activations = s_activations[0, -2, :].detach().cpu() -> I'm not actually sure what this means but I plan to actually run the code before the meeting on Sunday to fully understand
        - you add the activations to the diffs list to keep track of all the activations at all the layers for all the samples
    - for the second inner loop you go through all the layers of the non-sycophantic data and get the last activations
        - n_activations = n_activations[0, -2, :].detach().cpu() -> I'm not actually sure what this means but I plan to actually run the code before the meeting on Sunday to fully understand
    - ![Alt text](file:///Users/shizacharania/Downloads/IMG_1521.jpg) - diffs[layer][-1] -= n_activations I don't get this either
- then we layer all the layers + save them with their according file names

In [138]: I don't get the file size either so I will run the code later and figure that out

In [266]: passing in truthful QA and formatting similar to the other datasets

In [471]: take the average of all the layers (I believe these are the sycophanatic ones) -> normalize it THIS IS THE STEERING VECTOR. also we have selected layer 28 to test

In [484]: reset all activations (we already kept them on track) and on layer 28 (120 and -120 are kinda like the eigenvalues)
- "I added a multiple of the normalized steering vector to elicit more sycophantic outputs or subtracted a multiple to elicit less sycophantic outputs.”
    - multiply 120 by the unit normalized and avg vector we got (this is the sycophantic output)
    - multiply by -120 by the unit normalized and avg vector we got (this is the non sycophantic output)
- append a dictionary into the results array with the sycophantic, non-sycophnatic answers, and the sentence it outputs to know the differences

In [417]: they do the same thing but with layer 28

for both layers, they save the results in an external .json file