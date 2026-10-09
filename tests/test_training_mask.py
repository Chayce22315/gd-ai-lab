from gdai.train import collate_examples
from gdai.tokenizer import BOS_ID, TASK_ID, RESPONSE_ID, EOS_ID


def test_prompt_loss_is_masked_and_answer_is_trained():
    # Full stream: BOS, TASK, prompt bytes, RESPONSE, answer bytes, EOS
    tokens = [BOS_ID, TASK_ID, ord('p'), RESPONSE_ID, ord('{'), ord('}'), EOS_ID]
    x, y = collate_examples([(tokens, 4)])
    # targets are positions 1..end; prompt and RESPONSE are masked,
    # first target byte '{' starts at y index response_start-1 == 3.
    assert (y[0, :3] == -100).all()
    assert y[0, 3].item() == ord('{')
    assert y[0, -1].item() == EOS_ID
