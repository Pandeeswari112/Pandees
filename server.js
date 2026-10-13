const express = require('express');
const fs = require('fs');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

app.post('/api/book-appointment', (req, res) => {
    const newData = req.body;
    console.log("New Appointment:", newData);
    
    let contacts = [];
    if (fs.existsSync('contacts.json')) {
        const fileData = fs.readFileSync('contacts.json', 'utf-8');
        if(fileData) contacts = JSON.parse(fileData);
    }
    contacts.push(newData);
    fs.writeFileSync('contacts.json', JSON.stringify(contacts, null, 2));
    res.json({ success: true, message: "Booked!" });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'contact.html'));
});

app.listen(PORT, () => {
    console.log(`✅ Server running at http://localhost:${PORT}`);
});