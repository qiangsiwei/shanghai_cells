# -*- encoding:utf-8 -*-

def to_unicode(string):
	if isinstance(string, unicode):
		return string
	elif isinstance(string, str):
		return str(string).decode('utf-8')
	elif isinstance(string, float):
		return str(string).decode('utf-8')
	else:
		return u''
