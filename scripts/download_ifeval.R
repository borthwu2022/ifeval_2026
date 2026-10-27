
# install.packages("jsonlite")

# dir.create("data", showWarnings = FALSE)

url <- paste0(
  "https://huggingface.co/datasets/google/IFEval/",
  "resolve/main/ifeval_input_data.jsonl"
)

download.file(
  url,
  destfile = "data_raw/ifeval_input_data.jsonl",
  mode = "wb"
)

library(jsonlite)

prompts <- stream_in(
  file("data_raw/ifeval_input_data.jsonl"),
  verbose = FALSE
)

nrow(prompts)      # 541
names(prompts)
head(prompts$key)

