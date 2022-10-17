"""
This module ensures that the progress of
the model is printed and then saved in a nice
readable way.
"""

from rich.console import Console
import datetime

from ._rich_themes import SCRIPT_INFO

class ScriptInformation:

  def __init__(self):
      self.theme = SCRIPT_INFO
      self.c = Console(theme=self.theme, record=True)
      self.time_format = "%B %d %Y %H:%M:%S"

  def section_start(self, text):
      self.c.rule(text, style="section")

  def working_on(self, text):
      self.c.print(":snake: " + text, style="working")

  def script_time(self):
      text = datetime.datetime.now().strftime(self.time_format)
      self.c.print(":calendar: " + text, style="time")

  def important_metric(self, desc, metric):
      text = f":traffic_light: [imp_desc]{desc}[/]: [imp_metr]{metric}[/]"
      self.c.print(text)

  def author(self, firstn, lastn):
      text = f":panda_face: [subtle]Created by[/] [foam]{firstn} {lastn}[/]"
      self.c.print(text)

  def save(self, path):
      self.c.save_html(path)

