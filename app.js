const express = require('express');
var base64ToImage = require('base64-to-image');
const app = express();
var cors = require('cors')
app.use(cors())

const { spawn } = require('child_process');
const bodyParser = require('body-parser');
app.use(bodyParser.json({ limit: '10mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static('public'));
app.use('/images', express.static(__dirname + '/images'));
app.set('view engine', 'ejs');
app.get('/', (req, res) => {
    res.render('index');
})

app.post('/save_image', (req, res) => {

    const base64URI = req.body.base64Src;
    var base64Str = base64URI;
    var path = './public/images/';
    var optionalObj = { 'fileName': 'old', 'type': 'jpg' };

    base64ToImage(base64Str, path, optionalObj);

    var imageInfo = base64ToImage(base64Str, path, optionalObj);

   
    const pyProg = spawn('python3', ['deadliftApp.py']);

    res.json({ message: "Image Saved Successfully", imageInfo: imageInfo });
});

app.listen(3000);