import { describe, it, expect } from 'vitest'
import { reactive } from 'vue'

function makeRegisterForm() {
  const form = reactive({ username: '', email: '', password: '', confirm: '' })
  const errors = reactive({ username: '', email: '', password: '', confirm: '' })

  function validate(): boolean {
    errors.username = ''
    errors.email = ''
    errors.password = ''
    errors.confirm = ''

    if (!form.username || form.username.length < 3) {
      errors.username = 'Username must be at least 3 characters'
    }

    if (!form.email) {
      errors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(form.email)) {
      errors.email = 'Enter a valid email'
    }

    if (!form.password || form.password.length < 8) {
      errors.password = 'Password must be at least 8 characters'
    } else if (!/[A-Z]/.test(form.password)) {
      errors.password = 'Must contain at least one uppercase letter'
    } else if (!/[0-9]/.test(form.password)) {
      errors.password = 'Must contain at least one number'
    }

    if (!form.confirm || form.password !== form.confirm) {
        errors.confirm = 'Passwords do not match'
    }

    return !errors.username && !errors.email && !errors.password && !errors.confirm
  }

  return { form, errors, validate }
}

function validForm() {
  return {
    username: 'johndev',
    email: 'john@example.com',
    password: 'Secret123',
    confirm: 'Secret123',
  }
}

describe('RegisterView: validate()', () => {
  describe('username', () => {
    it('fails when username is empty', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), username: '' })

      expect(validate()).toBe(false)
      expect(errors.username).toBe('Username must be at least 3 characters')
    })

    it('fails when username is shorter than 3 characters', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), username: 'ab' })

      expect(validate()).toBe(false)
      expect(errors.username).toBe('Username must be at least 3 characters')
    })

    it('passes with exactly 3 characters', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), username: 'abc' })

      expect(validate()).toBe(true)
      expect(errors.username).toBe('')
    })
  })

  describe('email', () => {
    it('fails when email is empty', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), email: '' })

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Email is required')
    })

    it('fails when email format is invalid', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), email: 'bademail' })

      expect(validate()).toBe(false)
      expect(errors.email).toBe('Enter a valid email')
    })

    it('passes with valid email', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, validForm())

      expect(validate()).toBe(true)
      expect(errors.email).toBe('')
    })
  })

  describe('password', () => {
    it('fails when password is empty', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), password: '', confirm: '' })

      expect(validate()).toBe(false)
      expect(errors.password).toBe('Password must be at least 8 characters')
    })

    it('fails when password is shorter than 8 characters', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), password: 'Ab1', confirm: 'Ab1' })

      expect(validate()).toBe(false)
      expect(errors.password).toBe('Password must be at least 8 characters')
    })

    it('fails when password has no uppercase letter', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), password: 'alllower1', confirm: 'alllower1' })

      expect(validate()).toBe(false)
      expect(errors.password).toBe('Must contain at least one uppercase letter')
    })

    it('fails when password has no digit', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), password: 'NoDigitsHere', confirm: 'NoDigitsHere' })

      expect(validate()).toBe(false)
      expect(errors.password).toBe('Must contain at least one number')
    })

    it('passes with valid password', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, validForm())

      expect(validate()).toBe(true)
      expect(errors.password).toBe('')
    })
  })

  describe('confirm', () => {
    it('fails when confirm does not match password', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), confirm: 'Different1' })

      expect(validate()).toBe(false)
      expect(errors.confirm).toBe('Passwords do not match')
    })

    it('fails when confirm is empty but password is set', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, { ...validForm(), confirm: '' })

      expect(validate()).toBe(false)
      expect(errors.confirm).toBe('Passwords do not match')
    })

    it('passes when confirm matches password', () => {
      const { form, errors, validate } = makeRegisterForm()
      Object.assign(form, validForm())

      expect(validate()).toBe(true)
      expect(errors.confirm).toBe('')
    })
  })

  describe('combined', () => {
    it('returns false and sets all errors when form is completely empty', () => {
      const { errors, validate } = makeRegisterForm()

      expect(validate()).toBe(false)
      expect(errors.username).toBeTruthy()
      expect(errors.email).toBeTruthy()
      expect(errors.password).toBeTruthy()
      expect(errors.confirm).toBeTruthy()
    })

    it('returns true when all fields are valid', () => {
      const { form, validate } = makeRegisterForm()
      Object.assign(form, validForm())

      expect(validate()).toBe(true)
    })

    it('clears previous errors on re-validation', () => {
      const { form, errors, validate } = makeRegisterForm()

      validate()
      Object.assign(form, validForm())
      validate()

      expect(errors.username).toBe('')
      expect(errors.email).toBe('')
      expect(errors.password).toBe('')
      expect(errors.confirm).toBe('')
    })
  })
})