from lxml import etree
from urllib import request
import sh

url_dir_base = 'http://www.southparkwillie.com/Season%d/'
dialog_dir_tmpl = 'dialogs/season-%d-%s'
dialog_file_tmpl = dialog_dir_tmpl + '/%d.tsv'

def commit(season, script_page, num, speakers, utterances):
    if len(speakers) == 0 or len(utterances) == 0:
        return
    sh.mkdir('-p', dialog_dir_tmpl % (season, script_page))
    filename = dialog_file_tmpl % (season, script_page, num)
    print('\t\t%s %d' % (filename, len(speakers)))
    with open(filename, 'w') as f:
        for speaker, utterance in zip(speakers, utterances):
            f.write('\t'.join(['', speaker, '', utterance]) + '\n')

num = 0
for season in range(1, 21):
    print('Season %d' % season)

    url_dir = url_dir_base % season
    with request.urlopen(url_dir) as f:
        doc = etree.parse(f, parser=etree.HTMLParser())

    links = doc.findall('body/ul/li/a')
    pages = [link.attrib['href'] for link in links]
    script_pages = [p for p in pages if p.endswith('script.htm')]

    for script_page in script_pages:
        print('\tScript %s' % script_page)
        script_url = url_dir + '/' + script_page
        with request.urlopen(script_url) as f:
            text = f.read().decode('windows-1252')

        rows = text.split('<tr>')
        speakers = []
        utterances = []
        num = 0
        for row in rows:
            rowtext = ' '.join(row.split('\r\n'))
            rowdoc = etree.fromstring(rowtext, parser=etree.HTMLParser())

            cells = rowdoc.findall('body/td')
            if len(cells) == 2:
                speaker_td, utterance_td = cells
                speaker = ' '.join(t.strip() for t in speaker_td.itertext())
                if len(speaker) != 0 or speaker.find(':') != -1:
                    utterance = ' '.join(u.strip() for u in utterance_td.itertext())
                    colon = speaker.find(':')
                    speaker = speaker[:speaker.find(':')]
                    if len(speaker) > 1:
                        speakers.append(speaker)
                        utterances.append(utterance)
                        continue
            commit(season, script_page, num, speakers, utterances)
            speakers = []
            utterances = []
            num += 1
        commit(season, script_page, num, speakers, utterances)
