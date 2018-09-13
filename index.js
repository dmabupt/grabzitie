const http = require('http');
var fs = require('fs');
const puppeteer = require('puppeteer');

const tagDivG = /<div class="imgInfo fl">[\s\S]+?<\/div>/g;
const tagImgG = /<img src="([\s\S]+?)" width/;
const tagLabelG = /imgAuthor">([\s\S]+?)<\/p>/;

const ziti = ['k', 'x', 'c'];
const text = ['的', '一', '国', '在', '人', '了', '有', '中', '是', '年', '和', '大', '业', '不', '为', '发', '会', '工', '经', '上', '地', '市', '要', '个', '产', '这', '出', '行', '作', '生', '家', '以', '成', '到', '日', '民', '来', '我', '部', '对', '进', '多', '全', '建', '他', '公', '开', '们', '场', '展', '时', '理', '新', '方', '主', '企', '资', '实', '学', '报', '制', '政', '济', '用', '同', '于', '法', '高', '长', '现', '本', '月', '定', '化', '加', '动', '合', '品', '重', '关', '机', '分', '力', '自', '外', '者', '区', '能', '设', '后', '就', '等', '体', '下', '万', '元', '社', '过', '前', '面'];

function wait(ms=3000){
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function download(name, ziti, label) {
  const dir = 'download/' + ziti + '_' + label;
  if(fs.existsSync(dir) == false)
    fs.mkdirSync(dir);
  var dest = dir + '/' + name.replace(/\//g,'_');
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
      await page.goto(url, { waitUntil: 'load', timeout: 0 });
      await page.evaluate(() => { window.scrollBy(0, window.innerHeight); })
      // Get the height of the rendered page
      const bodyHandle = await page.$('body');
      const { height } = await bodyHandle.boundingBox();

      // Scroll one viewport at a time, pausing to let content load
      const viewportHeight = page.viewport().height;
      let viewportIncr = 0;
      while (viewportIncr + viewportHeight < height) {
        await page.evaluate(_viewportHeight => {
          window.scrollBy(0, _viewportHeight);
        }, viewportHeight);
        await wait(1000);
        viewportIncr = viewportIncr + viewportHeight;
      }
      
      // Scroll back to top
      await page.evaluate(_ => {
        window.scrollTo(0, 0);
      });

      // Some extra delay to let images load
      await wait(500);

      const bodyHTML = await page.evaluate(body => body.innerHTML, bodyHandle);
      await bodyHandle.dispose();
      // console.log(bodyHTML);
      const divs = bodyHTML.match(tagDivG);
      if(divs && divs.length > 0) {
        for(var i = 0; i < divs.length - 1; i++) {
          const img = divs[i].match(tagImgG);
          const label = divs[i].match(tagLabelG);
          if(img && img.length > 1 && label && label.length > 1 && label[1].length < 4) {
            await download(img[1], ziti[k], label[1]);
          }
        }
      }
    }
  }
  await browser.close();
};

run();
