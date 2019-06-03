from qqbot import QQBotSlot as qqbotslot,RunBot
from qqbot import qqbotsched as qqbotsched
from qqbot import _bot as bot
bot.Login(['-q', '1234'])
import time
@qqbotsched(hour='23',minute='01')
def mytask(bot):
    gl=bot.List('group','大逗比(~_~;)')
    if gl is not None:
        for group in gl:
            bot.SendTo(group,'请交日报')
def onQQMessage(bot,contact,member,content):
    if content.startswith("日报"):
        """
        打包我们的日志文件
        """
        bot.SendTo(contact,'日报已经收到')



if __name__=='__main__':
    RunBot()
    bot.lit('buddy')
