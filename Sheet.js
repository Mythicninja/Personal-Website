const scriptURL = 'https://script.google.com/macros/s/AKfycbwTzrD6p0NLzaybrvX5JTgf-285ZMOO8lxozSMUs971ZIsMY_9W1VuEFy8CPk1MmzneMA/exec'

const form = document.forms['contactForm']

form.addEventListener('submit', e=> {
    e.preventDefault()
    fetch(scriptURL, { method: 'POST', body: new FormData(form)})
    .then(() => { window.location.reload();})
    .catch(error => console.error('Error!', error.message))
})