# NLP_Done_Right

*Please cite this repository if you use any of the code in your research*

This repository is a demonstration on how to modularly write re-usable code for NLP for research/production. Kind of trying to bridge/merge my experiences in both academia and industry. A single codebase with many shared functions across several common NLP tasks. 

I hope you benefit from this to quickly spin up new peojects and not write everything from scratch.
Also please **NEVER** write flat python main.py files with 2000 lines of code ( A huge problem I have noticed in academia and also among many researchers in industry) .  

*Remember : Well writen code = Highly productive research*

**Framework** PyTorch

Addiitonally this contains Four Project Reports.   
http://www.cs.utexas.edu/~gdurrett/courses/fa2019/cs388.shtml.  

## Feature Engineered Named Entity Recognition
Named Entity Recognition(NER) is a fundamen-tal NLP task where the objective is to identify thenamed  entities  in  a  piece  of  text.   In  this  work,we focus on designing powerful features for NER,which leads to reasonably good accuracy withoutthe  need  of  a  complex  model.   We  tried  our  ap-proach on CONLL-2003 NER dataset which hasfour  class  of  named  entities:  person,  organiza-tion,  location,  and miscellaneous.  We just focuson identifying instances of the person label in iso-lation for this work.
