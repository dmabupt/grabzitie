const http = require('http');
var fs = require('fs');
const puppeteer = require('puppeteer');

const tagDivG = /<div class="imgInfo fl">[\s\S]+?<\/div>/g;
const tagImgG = /<img src="([\s\S]+?)" width/;
const tagLabelG = /imgAuthor">([\s\S]+?)<\/p>/;

const ziti = ['k', 'x', 'c'];
const text = ['的', '一', '国', '在', '人', '了', '有', '中', '是', '年', '和', '大', '业', '不', '为', '发', '会', '工', '经', '上', '地', '市', '要', '个', '产', '这', '出', '行', '作', '生', '家', '以', '成', '到', '日', '民', '来', '我', '部', '对', '进', '多', '全', '建', '他', '公', '开', '们', '场', '展', '时', '理', '新', '方', '主', '企', '资', '实', '学', '报', '制', '政', '济', '用', '同', '于', '法', '高', '长', '现', '本', '月', '定', '化', '加', '动', '合', '品', '重', '关', '机', '分', '力', '自', '外', '者', '区', '能', '设', '后', '就', '等', '体', '下', '万', '元', '社', '过', '前', '面'];

function sleep(ms=3000){
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function download(name, ziti, label) {
  if(fs.existsSync(ziti + '_' + label) == false)
    fs.mkdirSync(ziti + '_' + label);
  var dest = ziti + '_' + label + '/' + name.replace(/\//g,'_');
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
  const browser = await puppeteer.launch({ headless: false })
  const page = await browser.newPage();
  
  page.setViewport({
      width: 1200,
      height: 800
  })

  for(k in ziti) {
    for(j in text) {
      const url = 'http://www.saishufa.com/search.html?ziti=' + ziti[k] + '&word=' + encodeURI(text[j]);
      console.log(url);
      await page.goto(url, {waitUntil: 'load'});
      var scrollTimer = page.evaluate(() => {
        return new Promise((resolve, reject) => {
          var totalHeight = 0;
          var distance = 600;
          var timer = setInterval(() => {
            window.scrollBy(0, distance)
            totalHeight += distance;
            if(totalHeight >= document.body.scrollHeight){
              clearInterval(timer);
              resolve();
            }
          }, 500);
        })
      })

      var crawler = scrollTimer.then(async () => {
        var urls = await page.evaluate(() => {
          var links = [...document.querySelectorAll('div .imgInfo .fl')];
          return links.map(el => {
            const imgname = el.innerText.match(tagImgG);
            const labelname = el.innerText.match(tagLabelG);
            return {img: imgname, label: labelname};
          });
        })
        await page.close()
        return Promise.resolve(urls)
      }).catch((e) => {
        console.log(e);
      });

      crawler.then(urls => {
        console.log(urls.img + k + urls.label);
        // await download(urls.img, k, urls.label);
      });
    }
  }
  await browser.close();
};

run();
