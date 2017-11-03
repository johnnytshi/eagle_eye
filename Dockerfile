FROM johnnytshi/o_d:v1.0

COPY src/app.py /root/

WORKDIR /root
CMD python3 app.py \
    --source=$SOURCE \
    --slack_token=$SLACK_TOKEN \
    --slack_channel=$SLACK_CHANNEL \
    --min_score_thresh=$MIN_SCORE_THRESH
