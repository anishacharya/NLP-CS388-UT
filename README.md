# NLP-CS388-UT

*Please cite this repository if you use any of the code in your research*

This repository is a demonstration on how to modularly write re-usable code for NLP for research/production. Kind of trying to bridge/merge my experiences in both academia and industry. A single codebase with many shared functions across several common NLP tasks. 

I hope you benefit from this to quickly spin up new projects and not write everything from scratch.
Also please **NEVER** write flat python main.py files with 2000 lines of code ( A huge problem I have noticed in academia and also among many researchers in industry) .  

*Remember : Well writen code = Highly productive research*

**Framework** PyTorch

Additonally this contains Four Project Reports.   
http://www.cs.utexas.edu/~gdurrett/courses/fa2019/cs388.shtml.  


## Feature Engineered Named Entity Recognition
Named Entity Recognition(NER) is a fundamen-tal NLP task where the objective is to identify thenamed  entities  in  a  piece  of  text.   In  this  work,we focus on designing powerful features for NER,which leads to reasonably good accuracy withoutthe  need  of  a  complex  model.   We  tried  our  ap-proach on CONLL-2003 NER dataset which hasfour  class  of  named  entities:  person,  organiza-tion,  location,  and miscellaneous.  We just focuson identifying instances of the person label in iso-lation for this work.

## Named Entity Recognition as a sequence labeling problem
Named Entity Recognition(NER) is a fundamen-tal NLP task where the objective is to identify thenamed entities in a piece of text. In this work, wefocus on designing powerful features and modelsfor  NER,  which  leads  to  reasonably  good  accu-racy  without  the  need  of  a  complex  model.   Wetried our approach on CONLL-2003 NER datasetwhich has four class of named entities: person, or-ganization, location, and miscellaneous.

## Deep Ordered and Unordered Syntactic Compositionfor Sentiment Analysis
With  the  success  of  Deep  Learning  NLP  re-search has moved from shallow-models on sparse-feature-space   to   deep-models   on   dense   word-embedding-space.   In  this  work  we  explore  twopopular deep learning approaches on a sentimentclassification  task.    We  evaluate  our  models  onRotten Tomatoes movie reviews dataset, which as-signs a binary label to each movie review.

## Semantic Parsing as a Seq2Seq Problem
Semantic  parsing  is  the  task  of  translating  textto  a  formal  meaning  representation  such  as  log-ical  forms  or  structured  queries.   There  are  sev-eral decades of history associated with the classi-cal NLP task of semantic parsing.In  this  project  we  pose  the  problem  of  semanticparsing in a machine tranlation framework wherethe source is the standard text input while the tar-get  being  the  logical  form.   An  intuitive  way  tothink about it to think of logical form as anotherlanguage and then treat this as a translation prob-lem.In this setting, we use a sequence to sequence styleencoder-decoder architecture to do semnatic pars-ing. We particularly work on the geo-query datasetwhich in the downstream is used to do QA. Theinputs are plain english form questions while theoutputs/targets are logical forms that can be usedto query knowledge-graphs to get answers.

