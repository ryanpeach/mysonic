curl http://www.sonicgalaxy.net/wp-content/img/sprites/gen/sonic/ | \
ack -o '(?<=\").*.png(?=\")' | \
awk '$0="http://www.sonicgalaxy.net/wp-content/img/sprites/gen/sonic/"$0' | \
xargs wget
rm *zone*.png
rm *stage*.png
rm *font*.png
rm *logo*.png
rm *title*.png
rm huds.png
rm jukebox.png
rm misc-3.png
rm dr-robotnik-death.png
