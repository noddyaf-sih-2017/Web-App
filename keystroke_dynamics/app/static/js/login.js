let buffer = [] 
let oldstroke = new Object()
let oldstroke1 = new Object()
let newstroke = new Object()
let downbuffer =[]
let upbuffer =[]
let writebuf = []


const textField = document.getElementById('text')
const userField = document.getElementById('username')
let records= () =>
{
  let username = userField.value
  let password = textField.value

  if(!username){
    userField.focus()
    return false
  }
  else if(!password){
    password.focus()
    return false
  }

  if(writebuf.length > 0){
    qwest.post('/send_details/', {
      username: username,
      password: textField.value,
      json: JSON.stringify(writebuf)+','
    })
    .then((xhr, response)=>{
        writebuf  = []
        buffer = []
        downbuffer=[]
        upbuffer=[]

        if(response.authenticated){
          window.location.href = '/'
        }
        else{
          window.alert('Invalid password')
          textField.focus()
        }
    })
  }
  textField.value = ""
}

textField.onkeydown = (e) =>{
    let timestamp = Date.now() | 0
    let stroke = {
        key: e.key,
        keyCode: e.which,
        time: timestamp

    }
  
    if(stroke["key"] == "Enter"){
      records()
    }

    else if(stroke["key"] != "Shift" && stroke["key"] != "CapsLock"){
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
