from gdai.tokenizer import BOS_ID, EOS_ID, PAD_ID, RESPONSE_ID, TASK_ID, VOCAB_SIZE, decode_bytes, encode_bytes, encode_example


def test_byte_tokenizer_round_trip():
    text = 'Geometry Dash 2.2081: "glow" + 雨'
    assert decode_bytes(encode_bytes(text)) == text


def test_special_ids_do_not_overlap_bytes():
    assert max(BOS_ID, EOS_ID, PAD_ID, RESPONSE_ID, TASK_ID) < VOCAB_SIZE
    assert PAD_ID > 255


def test_example_keeps_response_start_and_eos():
    tokens, response_start = encode_example('task=deco', '{"ok":true}', 100)
    assert tokens[response_start:response_start + 2] == [ord('{'), ord('"')]
    assert tokens[-1] == EOS_ID
