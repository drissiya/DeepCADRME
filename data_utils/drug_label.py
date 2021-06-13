import os
import re
import sys
import time
from xml.etree import ElementTree
import numpy as np


class Label:
  def __init__(self, drug, track):
    self.drug = drug
    self.track = track
    self.sections = []
    self.mentions = []
    self.relations = []
    self.reactions = []

class Section:
  def __init__(self, id, name, text):
    self.id = id
    self.name = name
    self.text = text

class Mention:
  def __init__(self, id, section, type, start, len, str):
    self.id = id
    self.section = section
    self.type = type
    self.start = start
    self.len = len
    self.str = str
  def __str__(self):
    return 'Mention(id={},section={},type={},start={},len={},str="{}")'.format(
        self.id, self.section, self.type, self.start, self.len, self.str)
  def __repr__(self):
    return str(self)

class Relation:
  def __init__(self, id, type, arg1, arg2):
    self.id = id
    self.type = type
    self.arg1 = arg1
    self.arg2 = arg2
  def __str__(self):
    return 'Relation(id={},type={},arg1={},arg2={})'.format(
        self.id, self.type, self.arg1, self.arg2)
  def __repr__(self):
    return str(self)

class Reaction:
  def __init__(self, id, str):
    self.id = id
    self.str = str
    self.normalizations = []

class Normalization:
  def __init__(self, id, meddra_pt, meddra_pt_id, meddra_llt, meddra_llt_id, flag):
    self.id = id
    self.meddra_pt = meddra_pt
    self.meddra_pt_id = meddra_pt_id
    self.meddra_llt = meddra_llt
    self.meddra_llt_id = meddra_llt_id
    self.flag = flag
	
	
# Returns all the XML files in a directory as a dict
def xml_files(dir):
  files = {}
  for file in os.listdir(dir):
    if file.endswith('.xml'):
      files[file.replace('.xml', '')] = os.path.join(dir, file)
  return files


# Reads in the XML file
def read(file):
  root = ElementTree.parse(file).getroot()
  assert root.tag == 'Label', 'Root is not Label: ' + root.tag
  label = Label(root.attrib['drug'], root.attrib['track'])
  assert len(root) == 4, 'Expected 4 Children: ' + str(list(root))
  assert root[0].tag == 'Text', 'Expected \'Text\': ' + root[0].tag
  assert root[1].tag == 'Mentions', 'Expected \'Mentions\': ' + root[0].tag
  assert root[2].tag == 'Relations', 'Expected \'Relations\': ' + root[0].tag
  assert root[3].tag == 'Reactions', 'Expected \'Reactions\': ' + root[0].tag

  for elem in root[0]:
    assert elem.tag == 'Section', 'Expected \'Section\': ' + elem.tag
    label.sections.append(
        Section(elem.attrib['id'], \
                elem.attrib['name'], \
                elem.text))

  for elem in root[1]:
    assert elem.tag == 'Mention', 'Expected \'Mention\': ' + elem.tag
    label.mentions.append(
        Mention(elem.attrib['id'], \
                elem.attrib['section'], \
                elem.attrib['type'], \
                elem.attrib['start'], \
                elem.attrib['len'], \
                attrib('str', elem)))

  for elem in root[2]:
    assert elem.tag == 'Relation', 'Expected \'Relation\': ' + elem.tag
    label.relations.append(
        Relation(elem.attrib['id'], \
                 elem.attrib['type'], \
                 elem.attrib['arg1'], \
                 elem.attrib['arg2']))

  for elem in root[3]:
    assert elem.tag == 'Reaction', 'Expected \'Reaction\': ' + elem.tag
    label.reactions.append(
        Reaction(elem.attrib['id'], elem.attrib['str']))
    for elem2 in elem:
      assert elem2.tag == 'Normalization', 'Expected \'Normalization\': ' + elem2.tag
      label.reactions[-1].normalizations.append(
          Normalization(elem2.attrib['id'], \
                        attrib('meddra_pt', elem2), \
                        attrib('meddra_pt_id', elem2), \
                        attrib('meddra_llt', elem2), \
                        attrib('meddra_llt_id', elem2), \
                        attrib('flag', elem2)))

  return label

def attrib(name, elem):
  if name in elem.attrib:
    return elem.attrib[name]
  else:
    return None

