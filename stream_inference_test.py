from __future__ import print_function
from nmt.nmt import *  

args_in = "--src=de --tgt=en " \
    "--ckpt=/home/shuang/zhuiyi/tf/code/deen_gnmt_model_4_layer/translate.ckpt " \
    "--hparams_path=nmt/standard_hparams/wmt16_gnmt_4_layer.json " \
    "--out_dir=/tmp/deen_gnmt " \
    "--stream_infer " \
    "--vocab_prefix=/tmp/wmt16/vocab.bpe.32000 " \
    "--inference_ref_file=/tmp/wmt16/newstest2015.tok.bpe.32000.en ".split()


def stream_infer_test(args_in):
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    # FLAGS, unparsed = nmt_parser.parse_known_args(args_in)
    FLAGS = nmt_parser.parse_args(args_in)
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    inference_fn = inference.inference
    inference_stream = inference.stream_inference
    infer_model, loaded_infer_model, sess = run_main(FLAGS, default_hparams, train_fn, inference_fn)
    while True:
        text = raw_input("> ")
        print(inference_stream(infer_model, loaded_infer_model, sess, [text], default_hparams))


stream_infer_test(args_in)


