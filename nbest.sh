export CUDA_VISIBLE_DEVICES=3 #${X_SGE_CUDA_DEVICE}
#export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"
# python forward.py --function nbest --nbest nbest/dev.nbest.info.txt --model models/model.12.1L.pt 
# python forward.py --function nbest --nbest dev.small.info --model models/model.30.pt
# python forward.py --function nbest --nbest nbest/time_sorted_dev.nbestlist --model models/model.12.pt --lm rnn --rnnscale 6 --saveemb --cuda
# python forward.py --function nbest --nbest resnet_nbest/time_sorted_dev.nbestlist --model models/model.12.pt --lm rnn --ngram dev_ngram.st --rnnscale 16 --interp --factor 0.8 --cuda
# python forward.py --function nbest --nbest nbest/time_sorted_dev.nbestlist --model models/model.12.pt --lm default --cuda
# python forward_sep.py --function nbest --nbest resnet_nbest/time_sorted_dev.nbestlist --ngram resnet_nbest/time_sorted_Ldev.nbestlist --model models/L2model.12.hier.7.pt --lm curnn --rnnscale 16 --context '-5 -4 -3 -2 -1 1 2 3 4 5' --cuda
# python forward_backup.py --function nbest --nbest resnet_nbest/time_sorted_dev.nbestlist --model models/model.12.pt --lm rnn --ngram dev_ngram.st --rnnscale 12 --cuda
# python forward.py --function nbest --nbest resnet_nbest/time_sorted_dev.nbestlist --model models/L2model.12.pt --ngram resnet_nbest/time_sorted_Ldev.nbestlist --model2 models/model.12.pt --lm curnn --rnnscale 16 --context '-3 -2 -1 1 2 3' --cuda --interp --factor 0.8 --gscale 12.0
# python forward.py --function nbest --nbest rescore/time_sorted_dev.nbestlist --model models/model.12.1L.pt --lm rnn --rnnscale 16 --cuda --ngram rescore/time_sorted_Ldev.nbestlist --gscale 12.0 --factor 0.8
# python forward.py --function writeout --nbest resnet_nbest/time_sorted_dev.nbestlist --model models/model.12.1L.pt --lm rnn --rnnscale 16 --cuda --ngram resnet_nbest/time_sorted_Ldev.nbestlist --gscale 12.0 --factor 0.8
python forward.py --function nbest --nbest rescore/time_sorted_dev.nbestlist --model models/model.12.1L.pt --lm rnn --rnnscale 16 --cuda --logfile LOGs/nbestlog.txt
