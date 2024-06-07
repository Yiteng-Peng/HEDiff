import pyhelayers
import torch


def PredictPlainVector(plain_model, data):
    pred_p = plain_model(data)
    label_p = pred_p.argmax(1)
    return pred_p, label_p


def PredictEncVector(enc_model, data, context):
    # encode and encrypt
    iop = enc_model.create_io_processor()
    enc_data = pyhelayers.EncryptedData(context)
    iop.encode_encrypt_inputs_for_predict(enc_data,[data])
    # predictions
    enc_prediction = pyhelayers.EncryptedData(context)
    enc_model.predict(enc_prediction, enc_data)

    plain_prediction = iop.decrypt_decode_output(enc_prediction)
    pred_e = torch.tensor(plain_prediction)
    label_e = pred_e.argmax(1)
    return pred_e, label_e