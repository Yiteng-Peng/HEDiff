import torch


def PredictPlainVector(plain_model, data):
    pred_p = plain_model(data)
    label_p = pred_p.argmax(1)
    return pred_p, label_p


def PredictEncVector(enc_model, data, use_sim=True):
    fhe_mode = "simulate" if use_sim else "execute"
    pred_e = enc_model.forward(data.numpy(), fhe=fhe_mode)
    pred_e = torch.tensor(pred_e)
    label_e = pred_e.argmax(1)
    return pred_e, label_e