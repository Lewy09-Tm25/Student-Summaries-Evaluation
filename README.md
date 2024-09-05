# Student-Summaries-Evaluation

This project focuses on enabling automatic evaluation of a summary text written by a student on a specific topic. Let's jump in the files and directories involved here:-

## Project structure
1. /data - Consists datasets required for training and testing.
2. /notebooks - Project implementation

### /data
1. Summaries datasets - summaries_train.csv and summaries_test.csv contains the summaries of the students on the prompts. The students and prompts are represented by their ids. The last two columns are content and wording. These are the scores assigned by teachers. Our goal of automating the process of evaluating the summary text will have the metrics of content and wording as the target variables. The summaries are divided into training and testing sets, as the names suggest.
2. Prompts datasets - This entire project can be considered a prototype to something that can be evolved into a much larger scope. Hence, predictive modeling was done on 4 prompts only. These 4 prompts, their IDs, and the details of them like the prompt_question, and the prompt_text are given in these tables.


### /notebooks
Primary research and development pertaining to this project is done in this directory.


## Methodology
Here, we are dealing with 4 prompts and their questions. Hence, the prompts and the summaries tables are merged into one. This project can be implemented in 2 ways. First one is to merely one-hot encode the ids of the prompts in the main dataset and then follow the eda process. Another one is to develop a dedicated model specialized in scoring the student summary text of a particular prompt text. Since the former approach is fairly well-known and is an idea that is usually the first to cross across minds who are seeing this project for the first time, I decided to invest my time in the latter.
