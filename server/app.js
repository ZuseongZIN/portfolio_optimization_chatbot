var express = require('express');
const path = require('path');
const request = require('request');
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });
//console.log(path)
const TARGET_URL = 'https://api.line.me/v2/bot/message/reply'
const PUSH_TARGET_URL = 'https://api.line.me/v2/bot/message/push'
const TOKEN = process.env.TOKEN
const fs = require('fs');
const HTTPS = require('https');
const domain = process.env.domain
const sslport = 23023;
const bodyParser = require('body-parser');
var app = express();

var stockarr = [];
var pastreply = "";
var tatic ="";
var backtest = 0;
app.use('/simages', express.static(__dirname + '/src'));
app.use(bodyParser.json());
app.post('/hook', function (req, res) {
    var eventObj = req.body.events[0];
    var source = eventObj.source;
    var message = eventObj.message;
    // request log
    console.log('======================', new Date() ,'======================');
    console.log('[request]', req.body);
    console.log('[request source] ', eventObj.source);
    console.log('[request message]', eventObj.message);
    if(eventObj.type == 'postback')
    {   
        if(eventObj.postback.data == 'action=datetemp&selectId=1' && backtest == 0)
        {
            weight_recommend(eventObj.replyToken, eventObj.postback.params.date)
            stockarr.splice(0, stockarr.length);
            tatic = ""
            pastreply = ""
        }
        else if(eventObj.postback.data == 'action=datetemp&selectId=1' && backtest == 1)
        {
            optimizer(eventObj.source.userId, eventObj.postback.params.date)
            stockarr.splice(0, stockarr.length);
            tatic = ""
            pastreply = ""
            backtest = 0;
        }
    }
    else
    {   
        if(eventObj.message.text == '도움말' || eventObj.message.text == '주가 도움말' || eventObj.message.text == '비중 추천 도움말' || eventObj.message.text == '백테스트 도움말')
        {
            printhelp(eventObj.replyToken, eventObj.message.text)
        }
        else if (eventObj.message.text == '주가')
        {
            basicinform_pre(eventObj.replyToken, eventObj.message.text)
            pastreply = '주가'
        }
        else if ( pastreply == '주가' && eventObj.message.text.indexOf(' ') == -1){
            basicinform(eventObj.replyToken, eventObj.message.text)
        }
        else if (eventObj.message.text == '비중 추천'){
            weight_1(eventObj.replyToken, eventObj.message.text)
            pastreply = '비중 추천'
        }
        else if (eventObj.message.text == '백테스트'){
            weight_1(eventObj.replyToken, eventObj.message.text)
            pastreply = '비중 추천'
            backtest = 1
        }
        else if (pastreply == '비중 추천' && eventObj.message.text.indexOf(' ') != -1){
            var holder = eventObj.message.text.split(' ')
            var i;
            for(i = 0; i < holder.length; i++)
            {
                stockarr[i] = holder[i];
            }
            weight_2(eventObj.replyToken, eventObj.message.text);
            pastreply = "전략"
        }

        else if (pastreply == "전략" && (eventObj.message.text == 'gmv' || eventObj.message.text == 'ms') || eventObj.message.text == 'rp'){
            tatic = eventObj.message.text
            pastreply = "비중 추천"
            date(eventObj.replyToken, eventObj.message.text)
        }
        else {
            errormessage(eventObj.replyToken, eventObj.message.text);
            pastreply = ''
            backtest = 0
        }
    }
    
    res.sendStatus(200);
    
});

function errormessage(replyToken, message){
    request.post(
        {
            url: TARGET_URL,
            headers: {
                'Authorization': `Bearer ${TOKEN}`
            },
            json: {
                "replyToken":replyToken,
                "messages":[
                    {
                        "type":"text",
                        "text":"정해진 양식대로 입력하지 않으셨어요. \n 처음부터 다시 진행해주세요 :)"
                    }
                ]
            }
        },(error, response, body) => {
            console.log(body)
        });
}


function weight_1(replyToken, message){
    request.post(
        {
            url: TARGET_URL,
            headers: {
                'Authorization': `Bearer ${TOKEN}`
            },
            json: {
                "replyToken":replyToken,
                "messages":[
                    {
                        "type":"text",
                        "text":"포트폴리오에 넣을 종목을 입력해주세요. \n 두 종목 이상을 띄어쓰기로 구분해서 입력해주세요!"
                    }
                ]
            }
        },(error, response, body) => {
            console.log(body)
        });
}

function weight_2(replyToken, message){
    request.post(
        {
            url: TARGET_URL,
            headers: {
                'Authorization': `Bearer ${TOKEN}`
            },
            json: {
                "replyToken":replyToken,
                "messages":[
                    {
                        "type":"text",
                        "text":"전략을 입력해주세요 \n gmv, ms, rp 중 하나를 정확히 입력해주세요!"
                    }
                ]
            }
        },(error, response, body) => {
            console.log(body)
        });
}

function basicinform_pre(replyToken, message){
    request.post(
        {
            url: TARGET_URL,
            headers: {
                'Authorization': `Bearer ${TOKEN}`
            },
            json: {
                "replyToken":replyToken,
                "messages":[
                    {
                        "type":"text",
                        "text":"종목명을 입력해주세요!"
                    }
                ]
            }
        },(error, response, body) => {
            console.log(body)
        });
}

function printhelp(replyToken, message){
    if(message == '도움말'){
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {   
                            "type":"text",
                            "text":"궁금하신 기능을 선택해주세요.",
                            "quickReply": {
                                "items": [
                                  {
                                    "type": "action",
                                    "action": {
                                      "type": "message",
                                      "label": '주가 도움말',
                                      "text": '주가 도움말'
                                    }
                                  },
                                  {
                                    "type": "action",
                                    "action": {
                                      "type": "message",
                                      "label": '비중 추천 도움말',
                                      "text": '비중 추천 도움말'
                                    }
                                  },
                                  {
                                      "type": "action",
                                      "action": {
                                        "type": "message",
                                        "label": '백테스트 도움말',
                                        "text": '백테스트 도움말'
                                      }
                                  }
                                ]
                              }
                        }
                    ]
                }
            },(error, response, body) => {
                console.log(body)
            });
    }
    else if(message == '주가 도움말')
    {
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {
                            "type":"text",
                            "text":"'주가' 라고 입력하고 종목명을 입력해주세요. \n 현재가, 거래량, 전일대비 상승/하락률을 알려드릴게요!"
                        }
                    ]
                }
            },(error, response, body) => {
                console.log(body)
            });
    }
    else if(message == '비중 추천 도움말')
    {
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {
                            "type":"text",
                            "text":"'비중 추천'이라고 입력해주세요. \n 포트폴리오에 넣을 종목과 전략, 시작 날짜를 선택하면 비중을 추천해줄게요!"
                        }
                    ]
                }
            },(error, response, body) => {
                console.log(body)
            });
    }
    else if(message == '백테스트 도움말')
    {
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {
                            "type":"text",
                            "text":" '백테스트' 라고 입력해주세요. \n 포트폴리오에 넣을 종목과 전략, 시작 날짜를 선택하면 포트폴리오 수익률을 보여줄게요!"
                        }
                    ]
                }
            },(error, response, body) => {
                console.log(body)
            });
    }
    else if(message == '주가 도움말')
    {
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {
                            "type":"text",
                            "text":"사용자 : 주가\n챗봇: 종목명을 알려주세요.\n사용자: 종목명 입력 (ex 삼성전자)\n챗봇 : 현재가 거래량 전일대비 수익률"
                        }
                    ]
                }
            },(error, response, body) => {
                console.log(body)
            });
    }
    
}

function basicinform(replyToken, message) {
    var pystring;
    const spawn = require("child_process").spawn;
    const process = spawn("python", ["basic.py", message]);
    const Callback = (data) => {
        pystring = data.toString();
        if(pystring[0] == '1')
        {   pastreply =""
            pystring = pystring.replace('1현', '현');
            request.post(
                {
                    url: TARGET_URL,
                    headers: {
                        'Authorization': `Bearer ${TOKEN}`
                    },
                    json: {
                        "replyToken":replyToken,
                        "messages":[
                            {
                                "type":"text",
                                "text":pystring
                            }
                        ]
                    }
                },(error, response, body) => {
                    console.log(body)
                });
        }
        else
        {   
            var candarr = pystring.split('\n')
            request.post(
                {
                    url: TARGET_URL,
                    headers: {
                        'Authorization': `Bearer ${TOKEN}`
                    },
                    json: {
                        "replyToken":replyToken,
                        "messages":[
                            {
                                "type": "text",
                                "text": pystring,
                                "quickReply": {
                                  "items": [
                                    {
                                      "type": "action",
                                      "action": {
                                        "type": "message",
                                        "label": candarr[0],
                                        "text": candarr[0]
                                      }
                                    },
                                    {
                                      "type": "action",
                                      "action": {
                                        "type": "message",
                                        "label": candarr[1],
                                        "text": candarr[1]
                                      }
                                    },
                                    {
                                        "type": "action",
                                        "action": {
                                          "type": "message",
                                          "label": candarr[2],
                                          "text": candarr[2]
                                        }
                                    }
                                  ]
                                }
                              }
                        ]
                        
                    }
                },(error, response, body) => {
                    console.log(body)
                });
        }
    };
    process.stdout.on("data", Callback);
    
}

function optimizer(userid, sdate) {
    sdate = sdate.toString();
    var pystring = ''
    var i;
    for(i = 0; i < stockarr.length; i++)
    {
        pystring += stockarr[i];
        pystring += ','
    }
    pystring += sdate;
    console.log(pystring);
    const spawn = require("child_process").spawn;
    const process = spawn("python", ["optimizer.py", pystring, 'backtest', tatic]);
    const Callback = (data) => {
        request.post(
            {
                url: PUSH_TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "to": userid,
                    "messages":[
                        {
                            "type":"text",
                            "text":'조회하신 주식의 백테스트 결과입니다.'
                        },
                        {
                            "type":"image",
                            "originalContentUrl": "https://" + domain+ ":23023/simages/test.png",
                            "previewImageUrl": "https://" + domain+ ":23023/simages/test.png"                    
                        }
                    ]
                }
            },(error, response, body) => {
            console.log(body)
            });
    }
    process.stdout.on("data", Callback);
}

function weight_recommend(replyToken, sdate) {
    sdate = sdate.toString();
    var pystring = ''
    var i;
    for(i = 0; i < stockarr.length; i++)
    {
        pystring += stockarr[i];
        pystring += ','
    }
    pystring += sdate;
    console.log(pystring);
    const spawn = require("child_process").spawn;
    const process = spawn("python", ["optimizer.py", pystring ,"weight",tatic]);
    const Callback = (data) => {
        pystring = data.toString();
        pystring = pystring.slice(1,-2)
        pastreply == ""
        request.post(
            {
                url: TARGET_URL,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`
                },
                json: {
                    "replyToken":replyToken,
                    "messages":[
                        {
                            "type":"text",
                            "text": "비중 추천 결과입니다!"
                        },
                        {
                            "type":"text",
                            "text":pystring
                        }
                    ]
                }
            },(error, response, body) => {
            console.log(body)
            });
    }
    process.stdout.on("data", Callback);
}

function date(replyToken, message) {

    /*
    var holder = message.split(' ')
    var i;
    for(i = 0; i < holder.length; i++)
    {
        stockarr[i] = holder[i];
    }
    */
    var today = new Date();   
    var year = today.getFullYear();
    var month = today.getMonth() + 1;
    var date = today.getDate();
    if(month < 10)
    {
        month = '0'+ month  
    }
    if(date < 10)
    {
        date = '0'+ date  
    }
    var stoday = year + '-' + month + '-' + date;
    const messageObject = {
        "type": "template",
        "altText": "this is a buttons template",
        "template": {
            "type": "buttons",
            "title": "조회하실 날짜를 선택하세요.",
            "text": "선택하신 날짜에서 현재(오늘)까지 조회됩니다.",
            "actions": [
                {
                  "type": "datetimepicker",
                  "label": "날짜 선택",
                  "mode": "date",
                  "initial":"2020-01-01",
                  "max":stoday,
                  "min":"2010-01-01",
                  "data": "action=datetemp&selectId=1"
                },
                {
                  "type": "postback",
                  "label": "처음부터 다시할래요",
                  "data": "action=cancel&selectId=2"
                },
            ]
        }
    };
    request.post(
        {
            url: TARGET_URL,
            headers: {
                'Authorization': `Bearer ${TOKEN}`
            },
            json: {
                "replyToken":replyToken,
                "messages":[
                    // {
                    //     "type":"text",
                    //     "text":'조회하실 날짜를 선택하세요. 선택하신 날짜에서 현재까지 조회됩니다.',
                    //     "quickReply": {
                    //         "items": [
                    //             {
                    //             "type": "action",
                    //             "action": {
                    //             "type": "datetimepicker",
                    //             "label":"날짜 선택하기",
                    //             "data":"storeId=12345",
                    //             "mode":"date",
                    //             "initial":"2015-01-01",
                    //             "max":stoday,
                    //             "min":"2010-01-01"
                    //                 }
                    //             }
                    //         ]
                    //     }
                    // },
                    // {
                    //     "type":"text",
                    //     "text":req.body.postback.params
                    // }
                    messageObject
                ]
                
            }
        },(error, response, body) => {
        console.log(body)
        });
    
}

try {
    const option = {
      ca: fs.readFileSync('/etc/letsencrypt/live/' + domain +'/fullchain.pem'),
      key: fs.readFileSync(path.resolve(process.cwd(), '/etc/letsencrypt/live/' + domain +'/privkey.pem'), 'utf8').toString(),
      cert: fs.readFileSync(path.resolve(process.cwd(), '/etc/letsencrypt/live/' + domain +'/cert.pem'), 'utf8').toString(),
    };
  
    HTTPS.createServer(option, app).listen(sslport, () => {
      console.log(`[HTTPS] Server is started on port ${sslport}`);
    });
  } catch (error) {
    console.log('[HTTPS] HTTPS 오류가 발생하였습니다. HTTPS 서버는 실행되지 않습니다.');
    console.log(error);
  }
