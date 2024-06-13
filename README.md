## Grounding Gaps in Language Model Generations

This repository contains source code for the paper **Grounding Gaps in Language Model Generations** by [Omar Shaikh](https://oshaikh.com/), [Kristina Gligorić](https://kristinagligoric.github.io/), [Ashna Khetan](https://ashnakhetan.github.io/), [Matthias Gerstgrasser](https://matthias.gerstgrasser.net/), [Diyi Yang](https://cs.stanford.edu/~diyiy/index.html), and [Dan Jurafsky](https://web.stanford.edu/~jurafsky/). Feel free to reach out to [Omar Shaikh](https://oshaikh.com/) with any questions!

[[Paper]](https://arxiv.org/abs/2311.09144)

### *Abstract* 

Effective conversation requires common ground: a shared understanding between the participants. Common ground, however, does not emerge spontaneously in conversation. Speakers and listeners work together to both identify and construct a shared basis while avoiding misunderstanding. To accomplish grounding, humans rely on a range of dialogue acts, like clarification (What do you mean?) and acknowledgment (I understand.). However, it is unclear whether large language models (LLMs) generate text that reflects human grounding. To this end, we curate a set of grounding acts and propose corresponding metrics that quantify attempted grounding. We study whether LLM generations contain grounding acts, simulating turn-taking from several dialogue datasets and comparing results to humans. We find that -- compared to humans -- LLMs generate language with less conversational grounding, instead generating text that appears to simply presume common ground. To understand the roots of the identified grounding gap, we examine the role of instruction tuning and preference optimization, finding that training on contemporary preference data leads to a reduction in generated grounding acts. Altogether, we highlight the need for more research investigating conversational grounding in human-AI interaction.

### *Repository Structure*

Most of our experiments in the paper concern (1) simulating the next turn in a conversation, and (2) classifying grounding acts.

We have two major subfolders:

1) simulation
2) classification

First, we want to generate a set of simulated next-turn conversations with an LLM.

Run the bash script run_simulation.sh. You can replace the model argument flag with an open-source model, and use VLLM to simulate outputs for the Mistral model series too. Check out the VLLM repo for more information there!

Next, you want to classify outputs from the simulations and the ground-truth conversations. Here, you want to use the run_classification.sh script.

From both of these scripts, you'll get JSON files of grounding-acts labels for both GPT and human-human conversations.

### *How do I cite this work?* 

Feel free to use the following BibTeX entry.

**BibTeX:**

```tex
@misc{shaikh2024grounding,
      title={Grounding Gaps in Language Model Generations}, 
      author={Omar Shaikh and Kristina Gligorić and Ashna Khetan and Matthias Gerstgrasser and Diyi Yang and Dan Jurafsky},
      year={2024},
      eprint={2311.09144},
      archivePrefix={arXiv}
}
```