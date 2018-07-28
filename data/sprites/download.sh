curl http://www.sonicgalaxy.net/wp-content/img/sprites/gen/sonic/ | \
ack -o '(?<=\").*.png(?=\")' | \
awk '$0="http://www.sonicgalaxy.net/wp-content/img/sprites/gen/sonic/"$0' | \
xargs wget
