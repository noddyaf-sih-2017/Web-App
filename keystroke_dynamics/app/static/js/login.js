let buffer = [] 
let oldstroke = new Object()
let oldstroke1 = new Object()
let newstroke = new Object()
let downbuffer =[]
let upbuffer =[]
let writebuf = []
let invalid = false

const passwordToCheck = 'pass@123'
const textField = document.getElementById('text')
const userField = document.getElementById('username')
const submitButton = document.getElementById('save')

submitButton.addEventListener('click', (e)=>{
    records()
})

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

  if(password != passwordToCheck){
    window.location.reload()
  }

  if(invalid){
    window.location.reload()
  }

  if(writebuf.length > 0){
    qwest.post('/send_login_details/', {
      username: username,
      password: textField.value,
      json: JSON.stringify(writebuf)+',',
    })
    .then((xhr, response)=>{
        writebuf  = []
        buffer = []
        downbuffer=[]
        upbuffer=[]
        invalid=false

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
        time: timestamp

    }

    if(stroke["key"] == "Backspace"){
        invalid = true
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
        time: timestamp
    }

    if(stroke["key"] != "Backspace" && stroke["key"] != "Enter" && stroke["key"] != "Shift")
    {
        upbuffer.push(stroke)
        let up = upbuffer.shift()
        let down = downbuffer.shift()
        /*
        ft   = flight time
        kft = key press plus flight time
        time = key press time


        */

        
        let ftime=-(oldstroke.time-down.time)
        if (ftime<0)
        {
            ftime=0
        }
        if (buffer.length==0)
        {
            ftime=null
        }

        oldstroke=up

        let time = up.time-down.time
        let kftime= ftime+time
        
        let _stroke = new Object()
        _stroke.key=down.key
        _stroke.kftime=kftime
        
        _stroke.time=time
        _stroke.ftime=ftime
        
        buffer.push(_stroke)
        writebuf.push(_stroke)

    }
}
