
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent
from convlab2.util.analysis_tool.analyzer import Analyzer

from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure the best performing model based on the ConvLab-2 leaderboard (https://github.com/thu-coai/ConvLab-2?tab=readme-ov-file#end-to-end-performance-on-multiwoz)
# BERT nlu
sys_nlu = BERTNLU(
    mode='all',
    config_file='multiwoz_all_context.json',
    model_file='https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/bert_multiwoz_all_context.zip')  # noqa

# sys_nlu = SVMNLU(mode='sys')
# simple rule DST
sys_dst = RuleDST()
# rule policy
sys_policy = RulePolicy()
# template NLG
sys_nlg = TemplateNLG(is_user=False)
# assemble

sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg,
        name='sys')


# Note: due to the system agent not being able to handle multiple sessions, the server will only be able to handle one session at a time.

@app.route('/init-session', methods=['POST'])
def init_session():
    sys_agent.init_session()
    return 'Session initialized'


@app.route('/get-answer', methods=['POST'])
def get_answer():
    data = request.json
    user_message = data.get('user_message')
    if user_message is None:
        return jsonify({"error": "Session not found"}), 404
    chatbot_answer = sys_agent.response(user_message)
    is_finished = sys_agent.is_terminated()
    return jsonify({"chatbot_answer": chatbot_answer, "is_finished": is_finished})


if __name__ == '__main__':
    app.run(debug=True, port=8888)
