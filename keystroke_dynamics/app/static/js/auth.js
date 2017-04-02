let buffer = [] 
let oldstroke = new Object()
let oldstroke1 = new Object()
let newstroke = new Object()
let downbuffer =[]
let upbuffer =[]
let writebuf = []


const textField = document.getElementById('text')
const username= document.getElementById('username').value

let records= () =>
{
  console.debug('polling')
  // if(writebuf.length > 0){
    let sendData = 'nope'
    if(writebuf.length >= 50) {
        sendData = JSON.stringify(writebuf.slice(0, 50)) + ','
        writebuf = writebuf.slice(51)
    }

    console.debug('sending')
    qwest.post('/send_details/', {
      username: username,
      password: '',
      json: sendData,
    })
    .then((xhr, response)=>{
        if(response.wasEntered){
            writebuf  = []
            buffer = []
            downbuffer=[]
            upbuffer=[]
        }
        console.debug(response)
        if(!response.authenticated){
          window.alert('not authenticated')
        }
        
        poll()
    })
  // }
}

let poll = () => {
    setTimeout(records, 2000)
} 

poll()

textField.onkeydown = (e) =>{
    let timestamp = Date.now() | 0
    let stroke = {
        key: e.key,
        keyCode: e.which,
        time: timestamp

    }

    if(stroke["key"] !== "Enter" && stroke["key"] != "Shift" && stroke["key"] != "CapsLock"){
        downbuffer.push(stroke)
    }

}

textField.onkeyup = (e) => {
 
    let timestamp = Date.now() | 0
    let stroke = {
        key: e.key,
        keyCode: e.which,
        time: timestamp
    }

    if(stroke["key"] != "Enter" && stroke["key"] != "Shift")
    {
        upbuffer.push(stroke)
        let up = upbuffer.shift()
        let down = downbuffer.shift()
        /*
        ft   = flight time
        kft = key press plus flight time
        time = key press time
        */
        let ftime
        try{
            ftime=-(oldstroke.time-down.time)
        }
        catch(e){
            ftime = 0
        }

        if (ftime<0)
        {
            ftime=0
        }
        if (buffer.length==0)
        {
            ftime=null
        }


        let time = up.time-down.time
        let uutime=oldstroke.time-up.time;
        let dutime=oldstroke1.time-up.time;
        let ddtime=oldstroke1.time-down.time;
        let kftime= ftime+time
        
        let _stroke = new Object()
        _stroke.key=down.keyCode
        _stroke.kftime=kftime
       
        _stroke.time=time
        _stroke.ftime=ftime

        _stroke.uutime=(-uutime);
        _stroke.ddtime=(-ddtime);
        _stroke.dutime=(-dutime);
    
        buffer.push(_stroke)
        writebuf.push(_stroke)

        oldstroke=up
        oldstroke1 = down

    }
}
