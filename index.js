const http = require('http');
var fs = require('fs');
const puppeteer = require('puppeteer');

const tagDivG = /<div class="imgInfo fl">[\s\S]+?<\/div>/g;
const tagImgG = /<img src="([\s\S]+?)" width/;
const tagLabelG = /imgAuthor">([\s\S]+?)<\/p>/;

const ziti = ['k', 'x', 'c'];
const text = ['的', '一', '不', '是', '了', '人', '在', '有', '我', '他', '这', '为', '之', '来', '大', '以', '个', '中', '上', '们', '到', '说', '国', '和', '地', '也', '子', '时', '道', '出', '而', '要', '于', '就', '下', '得', '可', '你', '年', '生', '自', '会', '那', '后', '能', '对', '着', '事', '其', '里', '所', '去', '行', '过', '家', '十', '用', '发', '天', '如', '然', '作', '方', '成', '者', '多', '日', '都', '三', '小', '军', '二', '无', '同', '么', '经', '法', '当', '起', '与', '好', '看', '学', '进', '种', '将'];

async function download(name, label, cb) {
  if(fs.existsSync(label) == false)
    fs.mkdirSync(label);
  var dest = label + '/' + name.replace(/\//g,'_');
  console.log(dest);
  if(fs.existsSync(dest) == false) {
    var file = fs.createWriteStream(dest);
    var url = 'http://www.saishufa.com/' + name;
    console.log(url);
    http.get(url, (response) => {
      response.pipe(file);
      file.on('finish', () => { file.close(); });
    }).on('error', (err) => {  });
  }
};

async function run() {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  for(k in ziti) {
    for(j in text) {
      const url = 'http://www.saishufa.com/search.html?ziti=' + ziti[k] + '&word=' + encodeURI(text[j]);
      console.log(url);
      await page.goto(url, {timeout: 0});
      const bodyHandle = await page.$('body');
      const bodyHTML = await page.evaluate(body => body.innerHTML, bodyHandle);
      // console.log(bodyHTML);
      const divs = bodyHTML.match(tagDivG);
      if(divs && divs.length > 0) {
        for(var i = 0; i < divs.length - 1; i++) {
          const img = divs[i].match(tagImgG);
          const label = divs[i].match(tagLabelG);
          if(img && img.length > 1 && label && label.length > 1 && label[1].length < 4) {
            await download(img[1], label[1]);
          }
        }
      }
    }
  }
  await browser.close();
};

run();
